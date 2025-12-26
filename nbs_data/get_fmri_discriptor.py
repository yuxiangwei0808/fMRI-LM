import h5py
import os
import pickle
import pandas as pd
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
from pathlib import Path
from scipy.stats import zscore
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import nibabel as nib
import networkx as nx
from scipy import signal, stats
from nilearn.datasets import fetch_atlas_schaefer_2018
from brainspace.gradient import GradientMaps
from brainspace.gradient.kernels import compute_affinity

################# Functional Connectivity  ############################
def mean_fc(corr_matrix, idx1, idx2):
    vals = corr_matrix[np.ix_(idx1, idx2)]
    if idx1 == idx2:
        # upper triangle only for within-networkq
        vals = vals[np.triu_indices_from(vals, k=1)]
    return np.nanmean(vals)

def fc_discriptor_helper(fc_matrix, subj, sess, networks, net2idx):
    if np.isnan(fc_matrix).any():
        fc_matrix = np.nan_to_num(fc_matrix)
    subject_fc = {}
    for i, net1 in enumerate(networks):
        for j, net2 in enumerate(networks[i:], i):
            subject_fc[f"{net1}-{net2}"] = mean_fc(fc_matrix, net2idx[net1], net2idx[net2])

    subject_fc['subject_id'] = subj
    subject_fc['session_id'] = sess
    return subject_fc

def get_fc_descriptor(fc_data, batch_size=50):
    dir_name = Path(fc_data).parent
    output_csv = dir_name / 'fc_descriptors.csv'

    if not os.path.exists(output_csv):
        f = h5py.File(fc_data, 'r')
        data = f['fc']
        
        atlas = fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=2, yeo_networks=7)
        atlas_labels = atlas['labels']
        network_names = [lbl.decode('utf-8').split('_')[2] for lbl in atlas_labels]
        networks = sorted(set(network_names))
        
        net2idx = {net: [i for i, netname in enumerate(network_names) if netname == net]
                for net in networks}

        process_fn = partial(fc_discriptor_helper, networks=networks, net2idx=net2idx)
        headers_written = False

        for i in range(0, len(data), batch_size):
            sample_ids = [f'sample_{j:06d}' for j in range(i, min(i + batch_size, len(data)))]
            subjects = [f['metadata/subjects'][j].decode('utf-8') for j in range(i, min(i + batch_size, len(data)))]
            sessions = [f['metadata/sessions'][j].decode('utf-8') for j in range(i, min(i + batch_size, len(data)))]
            fc_matrices = [data[sid][:] for sid in sample_ids]

            with ProcessPoolExecutor(max_workers=1) as executor:
                results = list(executor.map(process_fn, fc_matrices, subjects, sessions))

            batched_df = pd.DataFrame(results)
            mode = 'w' if not headers_written else 'a'
            header = not headers_written

            batched_df.to_csv(output_csv, mode=mode, header=header, index=False)
            headers_written = True

            del results

        f.close()

    df = pd.read_csv(output_csv)

    # z-score normalization across subjects for each network pair
    network_pairs = [col for col in df.columns if col not in ['subject_id', 'session_id']]
    z_network_pairs = [c + '_z' for c in network_pairs]
    df[z_network_pairs] = df[network_pairs].apply(zscore)

    def describe_z(z):
        if z < -2:
            return f"markedly lower than average (z={z:+.2f})"
        elif z < -1:
            return f"somewhat reduced (z={z:+.2f})"
        elif z <= 1:
            return f"within typical range (z={z:+.2f})"
        elif z <= 2:
            return f"somewhat elevated (z={z:+.2f})"
        else:
            return f"markedly elevated (z={z:+.2f})"

    for idx, row in df.iterrows():
        descriptions = {col: describe_z(row[col]) for col in z_network_pairs}
        
        # show top 3 and bottom 3
        sorted_nets = sorted(descriptions.items(), key=lambda x: float(row[x[0]]))
        top3 = sorted_nets[-3:]
        bottom3 = sorted_nets[:3]

        summary_top, summary_bottom = [], []
        for (netpair, desc) in top3:
            summary_top.append(f"Connectivity between {netpair} is {desc}.")
        for (netpair, desc) in bottom3:
            summary_bottom.append(f"Connectivity between {netpair} is {desc}.")

        summary = f"Subject shows the following connectivity patterns:\n" + \
        "- Top 3 elevated connections:\n" + " ".join(summary_top) + "\n" + \
        "- Top 3 reduced connections:\n" + " ".join(summary_bottom)
        df.at[idx, 'summary'] = summary
    
    df.to_csv(output_csv, index=False)
    return df

################# Gradient  ############################
def gradient_descriptor_helper(gradients, network_labels, group_stats):
    """
    gradients: array (N parcels x K gradients), subject’s aligned gradients
    network_labels: array (N parcels,), e.g., Yeo-7 network per parcel
    group_stats: dict with group mean/std per gradient
    
    Returns: dict of descriptors
    """
    desc = {}
    for g in range(gradients.shape[1]):
        vals = gradients[:, g]
        
        # Range / spread
        desc[f"gradient{g+1}_range"] = vals.max() - vals.min()
        
        # Variance
        desc[f"gradient{g+1}_var"] = vals.var()
        
        # Z-score vs group
        desc[f"gradient{g+1}_range_z"] = (
            (desc[f"gradient{g+1}_range"] - group_stats[f"gradient_{g+1}_range_mean"]) /
            group_stats[f"gradient_{g+1}_range_std"]
        )
        
        # Network averages
        for net in np.unique(network_labels):
            net_mean = vals[network_labels == net].mean()
            desc[f"gradient{g+1}_{net}_mean"] = net_mean
    
    return desc

def z_to_text(z):
    if z > 1.0: return "higher than cohort average"
    if z < -1.0: return "lower than cohort average"
    return "within typical range"

def get_gradient_descriptor(fc_data, batch_size=50):
    """Process fMRI file and compute gradient descriptors, saving to CSV."""
    dir_name = Path(fc_data).parent
    output_csv = dir_name / 'gradient_descriptors.csv'
    
    atlas = fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=2, yeo_networks=7)
    atlas_labels = atlas['labels']
    network_names = [lbl.decode('utf-8').split('_')[2] for lbl in atlas_labels]

    if not os.path.exists(output_csv):
        f = h5py.File(fc_data, 'r')
        data = f['fc']

        # Compute group average functional connectivity matrix in batches
        group_fc_sum = None
        total_samples = len(data)

        for i in tqdm(range(0, total_samples, batch_size), desc="Computing group FC matrix"):
            end_idx = min(i + batch_size, total_samples)
            batch_matrices = [data[f'sample_{j:06d}'][:] for j in range(i, end_idx)]
            batch_mean = np.mean(batch_matrices, axis=0)
            
            if group_fc_sum is None:
                group_fc_sum = batch_mean * (end_idx - i)
            else:
                group_fc_sum += batch_mean * (end_idx - i)

        group_fc_matrix = group_fc_sum / total_samples

        gm_group = GradientMaps(n_components=3, approach='dm')
        group_affinity = compute_affinity(group_fc_matrix, kernel='cosine')
        gm_group.fit(group_affinity)

        # get group stats
        group_stats = {}
        for i in range(gm_group.n_components):
            vals = gm_group.gradients_[:, i]
            group_stats[f'gradient_{i+1}_range_mean'] = vals.max() - vals.min()
            group_stats[f'gradient_{i+1}_range_std'] = np.std(vals)

        headers_written = False

        # Process subjects in batches
        for i in range(0, len(data), batch_size):
            batch_results = []
            end_idx = min(i + batch_size, len(data))
            
            for j in range(i, end_idx):
                fc_matrix = data[f'sample_{j:06d}'][:]
                subj = f['metadata/subjects'][j].decode('utf-8')
                sess = f['metadata/sessions'][j].decode('utf-8')

                # Compute affinity matrix using cosine similarity
                affinity = compute_affinity(fc_matrix, kernel='cosine')

                gm = GradientMaps(n_components=3, approach='dm', alignment='procrustes')
                gm.fit(affinity, reference=gm_group.gradients_)

                # gradients = gm.gradients_
                # lambdas = gm.lambdas_ 

                aligned_gradients = gm.aligned_

                desc_dict = gradient_descriptor_helper(aligned_gradients, np.array(network_names), group_stats)

                z_g1 = desc_dict['gradient1_range_z']
                desc_g1 = f"Principal gradient range is {z_to_text(z_g1)} ({z_g1:+.1f} SD)"
                if z_g1 > 1.0 or z_g1 < -1.0:
                    desc_g1 += ", indicating " + f"{'stronger separation' if z_g1 > 0 else 'reduced separation'}" + f" between sensory and transmodal regions."

                z_g2 = desc_dict['gradient2_range_z']
                desc_g2 = f"Second gradient range is {z_to_text(z_g2)} ({z_g2:+.1f} SD)."
                if z_g2 > 1.0 or z_g2 < -1.0:
                    desc_g2 += " This may reflect " + f"{'stronger' if z_g2 > 0 else 'reduced'}" + " segregation among sensory modalities."

                z_g3 = desc_dict['gradient3_range_z']
                desc_g3 = f"Third gradient range is {z_to_text(z_g3)} ({z_g3:+.1f} SD)."
                if z_g3 > 1.0 or z_g3 < -1.0:
                    desc_g3 += " This may reflect " + f"{'stronger' if z_g3 > 0 else 'reduced'}" + " higher-order control/association distinctions."

                # Combine all descriptors into one row
                result_row = desc_dict.copy()
                result_row['subject_id'] = subj
                result_row['session_id'] = sess
                result_row['desc_g1'] = desc_g1
                result_row['desc_g2'] = desc_g2
                result_row['desc_g3'] = desc_g3
                
                batch_results.append(result_row)

            # Save batch to CSV
            batch_df = pd.DataFrame(batch_results)
            mode = 'w' if not headers_written else 'a'
            header = not headers_written

            batch_df.to_csv(output_csv, mode=mode, header=header, index=False)
            headers_written = True

            del batch_results

        f.close()

    # Read and return the saved CSV
    df = pd.read_csv(output_csv)

    # get the network-level descriptions
    network_dict = {'Cont': 'Control', 'Default': 'DefaultMode', 'DorsAttn': 'DorsalAttention', 'SalVentAttn': 'VentralAttention', 'Limbic': 'Limbic', 'SomMot': 'Somatomotor', 'Vis': 'Visual'}
    network_columns = [c for c in df.columns if any(net in c for net in np.unique(network_names))]
    z_network_columns = [c + '_z' for c in network_columns]
    df[z_network_columns] = df[network_columns].apply(zscore)

    target_columns = [c for c in z_network_columns if 'gradient1' in c]

    for idx, row in df.iterrows():
        net_gradients = {}
        for col in target_columns:
            z = row[col]
            net_name = network_dict[col.split('_')[1]]
            net_gradients[net_name] = z

        dev_nets_high = [f"{net}" for net in net_gradients if net_gradients[net] > 1.0]
        dev_nets_low = [f"{net}" for net in net_gradients if net_gradients[net] < -1.0]

        if not dev_nets_high and not dev_nets_low:
            net_descs = "All networks' principal gradient values are within typical range."
        else:
            net_descs = "Notable network principal gradient patterns include: "
            if dev_nets_high:
                net_descs += "Elevated principal gradient values in " + ", ".join(dev_nets_high) + ". "
            if dev_nets_low:
                net_descs += "Reduced principal gradient values in " + ", ".join(dev_nets_low) + "."

        df.at[idx, 'desc_network'] = net_descs
    
    df.to_csv(output_csv, index=False)
    return df

def get_gradient_descriptor_combine(df_path):
    df = pd.read_csv(df_path)
    descs = []
    for idx, row in df.iterrows():
        desc = f"The subject shows the following functional gradient patterns:\n" + \
        f"- {row['desc_g1']}\n" + \
        f"- {row['desc_g2']}\n" + \
        f"- {row['desc_g3']}\n" + \
        f"- {row['desc_network']}"
        descs.append(desc)
    df['summary'] = descs
    df.to_csv(df_path, index=False)
    return df

##################################### Graph ##############################
def get_graph_helper(fc_matrix, network_names, threshold=0.1):
    """
    Compute basic graph metrics from a functional connectivity (FC) matrix.
    
    Parameters
    ----------
    fc_matrix : np.ndarray
        Symmetric connectivity matrix (n_nodes x n_nodes), values = correlations.
    threshold : float
        Proportional threshold (keep top X% strongest connections).

    Returns
    -------
    metrics : dict
        Dictionary of graph metrics (global + node-level).
    """
    # 1. Remove diagonal
    np.fill_diagonal(fc_matrix, 0)

    # 2. Threshold: keep top X% of absolute connections
    n_nodes = fc_matrix.shape[0]
    n_edges = int(threshold * n_nodes * (n_nodes - 1) / 2)

    # Flatten and sort
    triu_idx = np.triu_indices(n_nodes, k=1)
    edge_weights = np.abs(fc_matrix[triu_idx])
    cutoff = np.sort(edge_weights)[-n_edges]  # edge weight cutoff

    # Binary adjacency (keep strongest edges)
    adj_binary = (np.abs(fc_matrix) >= cutoff).astype(int)

    # Weighted adjacency (keep weights but prune weak edges)
    adj_weighted = np.where(np.abs(fc_matrix) >= cutoff, np.abs(fc_matrix), 0)

    # 3. Convert to networkx graph (for clustering coefficient etc.)
    G = nx.from_numpy_array(adj_weighted)

    # --- Global metrics ---
    global_eff = nx.global_efficiency(G)
    # modularity using Louvain method
    communities = nx.community.louvain_communities(G, weight='weight', resolution=1.0)
    modularity = nx.community.modularity(G, communities, weight='weight')
    avg_clustering = nx.average_clustering(G, weight='weight')

    # --- Node-level metrics ---
    node_strength = np.sum(adj_weighted, axis=0)  # weighted degree per node
    # average to networks
    network_strength = {}
    for net in np.unique(network_names):
        idx = [i for i, netname in enumerate(network_names) if netname == net]
        network_strength[net] = np.mean(node_strength[idx])

    return {
        "global_efficiency": float(global_eff),
        "modularity": float(modularity),
        "avg_clustering": float(avg_clustering),
        **{f"strength_{net}": float(network_strength[net]) for net in network_strength}
        # "node_strength": node_strength,   # array, one value per node
        # "communities": communities        # module assignment per node
    }


def get_graph_descriptor(fc_data, batch_size=50):
    dir_name = Path(fc_data).parent
    output_csv = dir_name / 'graph_descriptors.csv'
    
    atlas = fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=2, yeo_networks=7)
    atlas_labels = atlas['labels']
    network_names = [lbl.decode('utf-8').split('_')[2] for lbl in atlas_labels]

    if not os.path.exists(output_csv):
        f = h5py.File(fc_data, 'r')
        data = f['fc']

        headers_written = False

        for i in range(0, len(data), batch_size):
            sample_ids = [f'sample_{j:06d}' for j in range(i, min(i + batch_size, len(data)))]
            subjects = [f['metadata/subjects'][j].decode('utf-8') for j in range(i, min(i + batch_size, len(data)))]
            sessions = [f['metadata/sessions'][j].decode('utf-8') for j in range(i, min(i + batch_size, len(data)))]
            fc_matrices = [data[sid][:] for sid in sample_ids]

            with ProcessPoolExecutor(max_workers=16) as executor:
                results = list(executor.map(get_graph_helper, fc_matrices, [network_names]*len(fc_matrices)))

            batched_df = pd.DataFrame(results)
            batched_df['subject_id'] = subjects
            batched_df['session_id'] = sessions

            mode = 'w' if not headers_written else 'a'
            header = not headers_written
            batched_df.to_csv(output_csv, mode=mode, header=header, index=False)
            headers_written = True

            del results

        f.close()

    df = pd.read_csv(output_csv)

    target_columns = [c for c in df.columns if c not in ['subject_id', 'session_id']]
    z_target_columns = [c + '_z' for c in target_columns]
    df[z_target_columns] = df[target_columns].apply(zscore)

    network_dict = {'Cont': 'Control', 'Default': 'DefaultMode', 'DorsAttn': 'DorsalAttention', 'SalVentAttn': 'VentralAttention', 'Limbic': 'Limbic', 'SomMot': 'Somatomotor', 'Vis': 'Visual'}
    network_columns = [c for c in df.columns if any(net in c for net in np.unique(network_names))]
    z_network_columns = [c for c in network_columns if c.endswith('_z')]

    for idx, row in df.iterrows():
        # ---- Modularity ----
        z = row['modularity_z']
        if z >= 3.0:
            mod_desc = f"The functional network organization shows extremely reduced modularity (z={z:+.2f}), reflecting a pronounced separation of functional subsystems."
        elif 2.0 <= z < 3.0:
            mod_desc = f"The functional network organization shows strongly reduced modularity (z={z:+.2f}), indicating more distinct segregation between functional systems."
        elif 1.0 <= z < 2.0:
            mod_desc = f"The functional network organization shows moderately reduced modularity (z={z:+.2f}), suggesting slightly stronger separation of functional subsystems."
        elif -1.0 < z < 1.0:
            mod_desc = f"The functional network organization shows typical modularity compared with the cohort."
        elif -2.0 < z <= -1.0:
            mod_desc = f"The functional network organization shows moderately elevated modularity (z={z:+.2f}), suggesting slightly weaker segregation between functional subsystems."
        elif -3.0 < z <= -2.0:
            mod_desc = f"The functional network organization shows strongly elevated modularity (z={z:+.2f}), indicating notably weaker separation of functional subsystems."
        else:  # z <= -3.0
            mod_desc = f"The functional network organization shows extremely elevated modularity (z={z:+.2f}), reflecting a marked loss of modular structure."

        # ---- Global Efficiency ----
        z = row['global_efficiency_z']
        if z >= 3.0:
            eff_desc = f"Whole-brain efficiency is extremely elevated (z={z:+.2f}), indicating very strong integrative communication across regions."
        elif 2.0 <= z < 3.0:
            eff_desc = f"Whole-brain efficiency is strongly elevated (z={z:+.2f}), suggesting robust integration across distributed brain systems."
        elif 1.0 <= z < 2.0:
            eff_desc = f"Whole-brain efficiency is moderately elevated (z={z:+.2f}), reflecting slightly enhanced global communication efficiency."
        elif -1.0 < z < 1.0:
            eff_desc = f"Whole-brain efficiency is within typical range."
        elif -2.0 < z <= -1.0:
            eff_desc = f"Whole-brain efficiency is moderately reduced (z={z:+.2f}), suggesting mildly impaired integrative communication."
        elif -3.0 < z <= -2.0:
            eff_desc = f"Whole-brain efficiency is strongly reduced (z={z:+.2f}), indicating marked inefficiency in global information exchange."
        else:  # z <= -3.0
            eff_desc = f"Whole-brain efficiency is extremely reduced (z={z:+.2f}), reflecting severely weakened integrative communication."

        # ---- Average Clustering ----
        z = row['avg_clustering_z']
        if z >= 3.0:
            clust_desc = f"Average clustering coefficient is extremely elevated (z={z:+.2f}), indicating unusually strong local connectivity patterns."
        elif 2.0 <= z < 3.0:
            clust_desc = f"Average clustering coefficient is strongly elevated (z={z:+.2f}), suggesting notably enhanced local clustering among neighboring nodes."
        elif 1.0 <= z < 2.0:
            clust_desc = f"Average clustering coefficient is moderately elevated (z={z:+.2f}), reflecting slightly stronger local connectivity."
        elif -1.0 < z < 1.0:
            clust_desc = f"Average clustering coefficient is within typical range."
        elif -2.0 < z <= -1.0:
            clust_desc = f"Average clustering coefficient is moderately reduced (z={z:+.2f}), suggesting mildly diminished local connectivity."
        elif -3.0 < z <= -2.0:
            clust_desc = f"Average clustering coefficient is strongly reduced (z={z:+.2f}), indicating marked reduction in local connectivity."
        else:  # z <= -3.0
            clust_desc = f"Average clustering coefficient is extremely reduced (z={z:+.2f}), reflecting severely weakened local connectivity patterns."

        net_strength = {}
        for col in z_network_columns:
            z = row[col]
            net_name = network_dict[col.split('_')[1]]
            net_strength[net_name] = z
        # get networks that have deviated strength
        dev_nets_high = [f"{net}" for net in net_strength if net_strength[net] > 1.0]
        dev_nets_low = [f"{net}" for net in net_strength if net_strength[net] < -1.0]
        if not dev_nets_high and not dev_nets_low:
            net_desc = "All networks' node strengths are within typical range."
        else:
            net_desc = "Notable network strength deviations include: "
            if dev_nets_high:
                net_desc += "Elevated strength in " + ", ".join(dev_nets_high) + ". "
            if dev_nets_low:
                net_desc += "Reduced strength in " + ", ".join(dev_nets_low) + "."

        df.at[idx, 'desc_modularity'] = mod_desc
        df.at[idx, 'desc_efficiency'] = eff_desc
        df.at[idx, 'desc_clustering'] = clust_desc
        df.at[idx, 'desc_network_strength'] = net_desc
    
    df.to_csv(output_csv, index=False)
    return df

def get_graph_descirptor_combine(df_path):
    df = pd.read_csv(df_path)
    descs = []
    for idx, row in df.iterrows():
        desc = f"The subject shows the following graph theoretical properties of brain connectivity:\n" + \
        f"- Graph Modularity: {row['desc_modularity']}\n" + \
        f"- Global Efficiency: {row['desc_efficiency']}\n" + \
        f"- Average Clustering Coefficient: {row['desc_clustering']}\n" + \
        f"- Node-level Strengths: {row['desc_network_strength']}"
        descs.append(desc)
    df['summary'] = descs
    df.to_csv(df_path, index=False)
    return df

##################################### ICA ############################
ica_network_mapping = {
    "Subcortical": list(range(5)),
    "Auditory": list(range(5, 7)),
    "Sensorimotor": list(range(7, 16)),
    "Visual": list(range(16, 25)),
    "Cognitive Control": list(range(25, 42)),
    "Default Mode": list(range(42, 49)),
    "Cerebellar": list(range(49, 53)),
}

def ica_tc_helper(tc, tr=2):
    # tc: shape (n_networks, n_timepoints)
    tc = signal.detrend(tc, axis=1)  # detrend along timepoints
    z_tc = stats.zscore(tc, axis=1)

    # --- 1. Amplitude / Mean loading ---
    abs_mean_amp = np.mean(np.abs(tc), axis=1)  # mean absolute amplitude per network

    # --- 2. Temporal variability ---
    temp_var = np.std(tc, axis=1)  # standard deviation per network

    # --- 3. Frequency content ---
    freqs, psd = signal.welch(z_tc, axis=1, nperseg=min(256, z_tc.shape[1]), fs=1/tr)  # power spectral density per network
    low_freq_mask = (freqs >= 0.01) & (freqs <= 0.1)
    low_freq_power = np.sum(psd[:, low_freq_mask], axis=1)
    high_freq_mask = (freqs > 0.1) & (freqs <= 0.25)
    high_freq_power = np.sum(psd[:, high_freq_mask], axis=1)
    spectral_ratio = low_freq_power / (high_freq_power + 1e-6)  # avoid div by zero

    # --- 4. Temporal coherence / autocorrelation ---
    def autocorr(x):
        result = np.correlate(x, x, mode='full')
        return result[result.size // 2:] / np.arange(len(x), 0, -1)
    autocorr_vals = np.array([autocorr(z_tc[i]) for i in range(z_tc.shape[0])])
    autocorr_lag1 = autocorr_vals[:, 1]  # lag-1 autocorrelation

    # --- 5. Transient detection (outlier spikes) ---
    outlier_counts = np.sum(np.abs(z_tc) > 3, axis=1)  # count of timepoints with |z| > 3
    outlier_freq = outlier_counts / z_tc.shape[1]  # proportion of outlier timepoints

    return {
        "abs_mean_amp": abs_mean_amp,
        "temp_var": temp_var,
        "spectral_ratio": spectral_ratio,
        "autocorr_lag1": autocorr_lag1,
        "outlier_freq": outlier_freq
    }

def get_ica_descriptor_0(root_path):
    # data prepare
    ref = h5py.File(root_path, 'r')

    if 'UKB' in root_path:
        dataset_name = 'UKB'
    elif 'ABCD' in root_path:
        dataset_name = 'ABCD'
        subjd2icaid = pickle.load(open('data/ABCD/abcd_get_sub_id_to_ica_id.pkl', 'rb'))
    else:
        raise ValueError("Dataset not recognized in root_path.")

    ica_tc, ica_fnc, ica_falff = [], [], []
    subjs, sesss = [], []
    for i in tqdm(range(len(ref['time_series']))):
        subj = ref['metadata/subjects'][i].decode('utf-8')
        sess = ref['metadata/sessions'][i].decode('utf-8')
        
        if dataset_name == 'UKB':
            tc = nib.load(f'/data/qneuromark/Results/ICA/UKB/{subj}/{sess}/func/NeuroMark1_sub01_timecourses_ica_s1_.nii').get_fdata()
            post_results = loadmat(f'/data/qneuromark/Results/ICA/UKB/{subj}/{sess}/func/NeuroMark1_postprocess_results/NeuroMark1_post_process_sub_001.mat')
        elif dataset_name == 'ABCD':
            if (subj, sess) not in subjd2icaid:
                print(f"Subject {subj} session {sess} not found in ICA ID mapping. Skipping.")
                continue
            ica_id = subjd2icaid[(subj, sess)]
            ica_id = f'ICA{ica_id:05d}'
            tc = nib.load(f'/data/neuromark2/Results/ICA/ABCD/R5/{ica_id}/ABCD_sub01_timecourses_ica_s1_.nii').get_fdata()
            post_results = loadmat(f'/data/neuromark2/Results/ICA/ABCD/R5/{ica_id}/ABCD_postprocess_results/ABCD_post_process_sub_001.mat')

        # average within each network
        tc_avg_network = np.zeros((tc.shape[0], len(ica_network_mapping)))
        for j, net in enumerate(ica_network_mapping):
            tc_avg_network[:, j] = np.mean(tc[:, ica_network_mapping[net]], axis=1)
        tc_avg_network = tc_avg_network.T

        tc_features = ica_tc_helper(tc_avg_network, tr=2)
        tc_features_mean = {f"{key}_ave": np.mean(tc_features[key]) for key in tc_features}
        ica_tc_features = {}
        for key in tc_features:
            for j, net in enumerate(ica_network_mapping):
                ica_tc_features[f"{net}_{key}"] = tc_features[key][j]
        # add the averaged features to the dict
        ica_tc_features.update(tc_features_mean)

        # average FNC within each network
        fnc = post_results['fnc_corrs'][0]
        fnc_avg_network = {}
        for net1 in ica_network_mapping:
            for net2 in ica_network_mapping:
                key = f"{net1}-{net2}"
                if key not in fnc_avg_network and f"{net2}-{net1}" not in fnc_avg_network:
                    vals = fnc[np.ix_(ica_network_mapping[net1], ica_network_mapping[net2])]
                    if net1 == net2:
                        vals = vals[np.triu_indices_from(vals, k=1)]
                    fnc_avg_network[key] = np.nanmean(vals)

        # average gfALFF within each network
        falff = post_results['fALFF'][0]  # shape (53,)
        falff_avg_network = {}
        for net in ica_network_mapping:
            vals = falff[ica_network_mapping[net]]
            falff_avg_network[net] = np.nanmean(vals)

        ica_tc.append(ica_tc_features)
        ica_fnc.append(fnc_avg_network)
        ica_falff.append(falff_avg_network)
        subjs.append(subj)
        sesss.append(sess)

    ica_tc = pd.DataFrame(ica_tc)
    ica_tc['subject_id'] = subjs
    ica_tc['session_id'] = sesss
    ica_tc.to_csv(f'data/{dataset_name}/fmri/descriptors/ica_tc.csv', index=False)

    ica_fnc = pd.DataFrame(ica_fnc)
    ica_fnc['subject_id'] = subjs
    ica_fnc['session_id'] = sesss
    ica_fnc.to_csv(f'data/{dataset_name}/fmri/descriptors/ica_fnc.csv', index=False)

    ica_falff = pd.DataFrame(ica_falff)
    ica_falff['subject_id'] = subjs
    ica_falff['session_id'] = sesss
    ica_falff.to_csv(f'data/{dataset_name}/fmri/descriptors/ica_falff.csv', index=False)

def describe_z(z):
    if z < -2:
        return f"markedly lower than average (z={z:+.2f})"
    elif z < -1:
        return f"somewhat reduced (z={z:+.2f})"
    elif z <= 1:
        return f"within typical range (z={z:+.2f})"
    elif z <= 2:
        return f"somewhat elevated (z={z:+.2f})"
    else:
        return f"markedly elevated (z={z:+.2f})"
        
def get_ica_descriptor_1(batch_size=50):
    dataset_name = 'ABCD'
    assert os.path.exists(f'data/{dataset_name}/fmri/descriptors/ica_fnc.csv') and os.path.exists(f'data/{dataset_name}/fmri/descriptors/ica_falff.csv') and os.path.exists(f'data/{dataset_name}/fmri/descriptors/ica_tc.csv'), "Run get_ica_descriptor_0() first."

    ica_tc = pd.read_csv(f'data/{dataset_name}/fmri/descriptors/ica_tc.csv')
    ica_fnc = pd.read_csv(f'data/{dataset_name}/fmri/descriptors/ica_fnc.csv')
    ica_falff = pd.read_csv(f'data/{dataset_name}/fmri/descriptors/ica_falff.csv')

    # z-score normalization across subjects for each metric
    target_columns_tc = [c for c in ica_tc.columns if c not in ['subject_id', 'session_id']]
    z_target_columns_tc = [c + '_z' for c in target_columns_tc]
    ica_tc[z_target_columns_tc] = ica_tc[target_columns_tc].apply(zscore)

    target_columns_fnc = [c for c in ica_fnc.columns if c not in ['subject_id', 'session_id']]
    z_target_columns_fnc = [c + '_z' for c in target_columns_fnc]
    ica_fnc[z_target_columns_fnc] = ica_fnc[target_columns_fnc].apply(zscore)

    # change column names to avoid conflict with tc
    ica_falff.columns = [c + '_falff' if c not in ['subject_id', 'session_id'] else c for c in ica_falff.columns]
    target_columns_falff = [c for c in ica_falff.columns if c not in ['subject_id', 'session_id']]
    z_target_columns_falff = [c + '_z' for c in target_columns_falff]
    ica_falff[z_target_columns_falff] = ica_falff[target_columns_falff].apply(zscore)

    df = ica_tc.merge(ica_fnc, on=['subject_id', 'session_id']).merge(ica_falff, on=['subject_id', 'session_id'])
    
    for idx, row in tqdm(df.iterrows()):
        # --- Temporal features ---
        # first get descriptions for averaged features
        global_desc_tc = []
        for key in ['abs_mean_amp_ave', 'temp_var_ave', 'spectral_ratio_ave', 'autocorr_lag1_ave', 'outlier_freq_ave']:
            z = row[key + '_z']
            key = key.replace('_ave', '')
            if z > 2.0:
                phrase = {
                "abs_mean_amp": "exceptionally strong network engagement overall",
                "temp_var": "markedly volatile fluctuations across networks",
                "spectral_ratio": "pronounced dominance of slow oscillations",
                "autocorr_lag1": "very persistent temporal structure",
                "outlier_freq": "intense bursts occurring frequently"
            }[key]
            elif z > 1.0:
                phrase = {
                "abs_mean_amp": "stronger network engagement overall",
                "temp_var": "highly dynamic fluctuations across networks",
                "spectral_ratio": "predominance of slow oscillations",
                "autocorr_lag1": "persistent temporal structure",
                "outlier_freq": "frequent transient bursts"
            }[key]
            elif z < -2.0:
                phrase = {
                "abs_mean_amp": "substantially reduced activity across networks",
                "temp_var": "unusually stable temporal dynamics",
                "spectral_ratio": "pronounced shift toward faster oscillations",
                "autocorr_lag1": "very rapid state transitions",
                "outlier_freq": "rare transient events"
            }[key]
            elif z < -1.0:
                phrase = {
                "abs_mean_amp": "globally reduced activity",
                "temp_var": "remarkably stable temporal dynamics",
                "spectral_ratio": "faster oscillatory patterns",
                "autocorr_lag1": "rapid state transitions",
                "outlier_freq": "few transient events"
            }[key]
            else:
                phrase = {
                "abs_mean_amp": "typical overall network engagement",
                "temp_var": "typical temporal variability",
                "spectral_ratio": "balanced oscillatory dynamics",
                "autocorr_lag1": "typical temporal coherence",
                "outlier_freq": "typical transient activity"
            }[key]
                
            if key == 'spectral_ratio_ave':  # add the raw value for spectral ratio
                phrase += f" (spectral ratio={row[key]:.2f})"
            if key == 'outlier_freq_ave':
                phrase += f" (outlier frequency={row[key]:.2%})"
                
            global_desc_tc.append(phrase)
        global_desc_tc = "Overall, the ICA timecourses show " + ", ".join(global_desc_tc).rstrip(',') + '.'

        # network-level, get top1 and bottom1 for each metric
        net_descs_tc = []
        metric_dict = {'abs_mean_amp': 'absolute mean amplitude', 'temp_var': 'temporal variability', 'spectral_ratio': 'spectral ratio', 'autocorr_lag1': 'lag-1 autocorrelation', 'outlier_freq': 'outlier frequency'}
        for metric in ['abs_mean_amp', 'temp_var', 'spectral_ratio', 'autocorr_lag1', 'outlier_freq']:
            metric_cols = [c for c in z_target_columns_tc if c.endswith(f"_{metric}_z")]
            net_z = {c.split('_')[0]: row[c] for c in metric_cols}
            top_net = max(net_z, key=net_z.get)
            bottom_net = min(net_z, key=net_z.get)

            phrase_top = {
                "abs_mean_amp": "strong engagement",
                "temp_var": "dynamic fluctuations",
                "spectral_ratio": "slow oscillations",
                "autocorr_lag1": "persistent activity",
                "outlier_freq": "frequent transient bursts"
            }[metric]
            phrase_bottom = {
                "abs_mean_amp": "reduced activity",
                "temp_var": "stable dynamics",
                "spectral_ratio": "faster oscillations",
                "autocorr_lag1": "rapid state transitions",
                "outlier_freq": "few transient events"
            }[metric]

            metric_name = metric_dict[metric]
            if net_z[top_net] > 1.0:
                net_descs_tc.append(f"For {metric_name}, {top_net} network shows {phrase_top}.")
                if net_z[bottom_net] < -1.0:
                    net_descs_tc.append(f"In contrast, {bottom_net} network shows {phrase_bottom}.")
            elif net_z[bottom_net] < -1.0:
                net_descs_tc.append(f"For {metric_name}, {bottom_net} network shows {phrase_bottom}.")

        net_descs_tc = " ".join(net_descs_tc) if net_descs_tc else "No networks show marked deviations in temporal features."
        df.at[idx, 'desc_tc'] = global_desc_tc + " In terms of network-level features, " + net_descs_tc

        # --- FNC features ---
        fnc_descs = []
        fnc_cols = [c for c in z_target_columns_fnc if c.endswith('_z')]
        fnc_z = {c: row[c] for c in fnc_cols}
        top_fnc = sorted(fnc_z.items(), key=lambda x: x[1], reverse=True)[:3]
        bottom_fnc = sorted(fnc_z.items(), key=lambda x: x[1])[:3]

        ele_net = []
        for (netpair, z) in top_fnc:
            netpair = netpair.split('_')[0]
            netpair = netpair.replace('-', ' and ')
            if z > 1.0:
                ele_net.append(netpair)
        fnc_descs.append(f"Elevated connectivity between {', '.join(ele_net)}.") if ele_net else None
        if fnc_descs: fnc_descs[-1] = fnc_descs[-1].rstrip('.')
        red_net = []
        for (netpair, z) in bottom_fnc:
            netpair = netpair.split('_')[0]
            netpair = netpair.replace('-', ' and ')
            if z < -1.0:
                red_net.append(netpair)
        fnc_descs.append(f"Reduced connectivity between {', '.join(red_net)}.") if red_net else None
        if fnc_descs: fnc_descs[-1] = fnc_descs[-1].rstrip('.')

        fnc_desc = "Functional network connectivity shows the following notable patterns: " + "; ".join(fnc_descs) + "." if fnc_descs else "Functional network connectivity is within typical range."
        df.at[idx, 'desc_fnc'] = fnc_desc

        # --- fALFF features ---
        falff_descs = []
        falff_cols = [c for c in z_target_columns_falff if c.endswith('_z')]
        falff_z = {c: row[c] for c in falff_cols}
        top_falff = max(falff_z, key=falff_z.get)
        bottom_falff = min(falff_z, key=falff_z.get)

        top_net_name = top_falff.split('_')[0]
        bottom_net_name = bottom_falff.split('_')[0]

        if falff_z[top_falff] > 2.0:
            falff_descs.append(f"{top_net_name} network shows markedly elevated fALFF, indicating pronounced expression relative to the cohort.")
        elif falff_z[top_falff] > 1.0:
            falff_descs.append(f"{top_net_name} network shows elevated fALFF, indicating greater-than-cohort expression.")
        if falff_z[bottom_falff] < -2.0:
            falff_descs.append(f"{bottom_net_name} network shows markedly reduced fALFF, indicating notably diminished contribution compared with the cohort.")
        elif falff_z[bottom_falff] < -1.0:
            falff_descs.append(f"{bottom_net_name} network shows reduced fALFF, indicating reduced contribution compared with the cohort.")
        falff_desc = " ".join(falff_descs) if falff_descs else "All networks' fALFF are within typical range, indicating balanced contributions."

        df.at[idx, 'desc_falff'] = falff_desc

    df.to_csv(f'data/{dataset_name}/fmri/descriptors/ica_descriptions.csv', index=False)
    return df

def get_ica_descriptor_combine(df_path='data/UKB/fmri/descriptors/ica_descriptions.csv'):
    df = pd.read_csv(df_path)
    descs = []
    for idx, row in df.iterrows():
        desc = f"The subject shows the following ICA-derived functional network characteristics:\n" + \
        f"- Temporal Features: {row['desc_tc']}\n" + \
        f"- Functional Network Connectivity: {row['desc_fnc']}\n" + \
        f"- Fractional ALFF: {row['desc_falff']}"
        descs.append(desc)
    df['summary'] = descs
    df.to_csv(df_path, index=False)
    return df


################### "self" descriptors (without group relative)   ########################
def get_fc_self_descriptor(fc_data, batch_size=50):
    dir_name = Path(fc_data).parent.parent
    output_csv = dir_name / 'descriptors' / 'fc_self_descriptors.csv'

    df = pd.read_csv(dir_name / 'descriptors' / 'fc_descriptors.csv')
        
    # zscore normalization against all connectivities across the brain, per-subject
    conn_columns = [c for c in df.columns if c not in ['subject_id', 'session_id', 'summary']]
    for idx, row in df.iterrows():
        conn_values = np.array(row[conn_columns].values, dtype=float)
        conn_values = np.nan_to_num(conn_values)  # Replace NaN with 0
        z_conn_values = zscore(conn_values)
        for i, col in enumerate(conn_columns):
            df.at[idx, col + '_z'] = z_conn_values[i]

        # get the prominent connections to form descriptor
        prominent_conns = []
        for i, col in enumerate(conn_columns):
            z = z_conn_values[i]
            conn_name = col.replace('_z', '')
            if z > 2.0:
                prominent_conns.append(conn_name)
        if prominent_conns:
            desc = "The subject shows markedly prominent functional connections at: " + ", ".join(prominent_conns) + "."
        else:
            desc = "No connectivities show markedly prominent."

        df.at[idx, 'summary'] = desc
    df.to_csv(output_csv, index=False)
    return df

sub_roi_mapping = {
    'Post': 'Postcentral gyrus',
    'FEF': 'Frontal eye field',
    'PrCv': 'Ventral precentral cortex',

    'ParOper': 'Parietal operculum',
    'TempOcc': 'Temporo-occipital cortex',

    'FrOperIns': 'Frontal operculum and insula',

    'Med': 'Medial cortex',
    'OFC': 'Orbitofrontal cortex',

    'TempPole': 'Temporal pole',

    'Par': 'Parietal cortex',
    'Temp': 'Temporal cortex',
    'PFC': 'Prefrontal cortex',

    'pCunPCC': 'Precuneus and posterior cingulate cortex',

    'TempOccPar': 'Temporo-occipital-parietal junction',
}

def get_region_self_descriptor(data_path):
    # describe the prominant regions based on roi data
    output_csv = 'data/UKB/fmri/descriptors/region_self_descriptors.csv'

    atlas = fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=2, yeo_networks=7)
    atlas_labels = atlas['labels']
    for i, lbl in enumerate(atlas_labels):
        lbl = lbl.decode('utf-8').split('_')
        lbl_name = f'{lbl[2]}'
        lbl_name += ' at right hemisphere' if lbl[1] == 'RH' else ' at left hemisphere'
        if sub_roi_mapping.get(lbl[3], None):
            lbl_name += f' {sub_roi_mapping[lbl[3]].lower()}'
        atlas_labels[i] = lbl_name

    out = []
    with h5py.File(data_path, 'r') as f:
        keys = list(f['time_series'].keys())
        subjs = [str(s.decode('utf-8')) for s in f['metadata/subjects'][:]]
        sesss = [str(s.decode('utf-8')) for s in f['metadata/sessions'][:]]

        for i, key in enumerate(tqdm(keys)):
            ts = f['time_series'][key][:]
            ts = ts[50:]  # remove first 50 rois (subcortex), only keep shaefer-400
            mean_ts = np.mean(ts, axis=1)  # mean time series per ROI
            z_mean_ts = zscore(mean_ts)

            prominent_rois = []
            for j in range(len(z_mean_ts)):
                z = z_mean_ts[j]
                if z > 2.0:
                    prominent_rois.append(atlas_labels[j].decode('utf-8'))
            if prominent_rois:
                desc = "The subject shows markedly prominent regional activity at: " + ", ".join(prominent_rois) + "."
            else:
                desc = "No regions show markedly prominent activity."

            out.append({
                'subject_id': subjs[i],
                'session_id': sesss[i],
                'summary': desc
            })

    df = pd.DataFrame(out)
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    fc_data = 'data/UKB/fmri/fc/data_fc.h5'
    # df = get_fc_descriptor(fc_data)

    # df = get_gradient_descriptor(fc_data)
    # get_gradient_descriptor_combine('data/UKB/fmri/descriptors/gradient_descriptors.csv')

    # df = get_graph_descriptor(fc_data)
    # get_graph_descirptor_combine('data/UKB/fmri/descriptors/graph_descriptors.csv')

    # get_ica_descriptor_0('data/ABCD/fmri/TianS3/data_resampled.h5')

    # get_ica_descriptor_1()
    # get_ica_descriptor_combine('data/ABCD/fmri/descriptors/ica_descriptions.csv')

    # df_fc_self = get_fc_self_descriptor(fc_data)
    # get_region_self_descriptor('data/UKB/fmri/TianS3/data_resampled.h5')
    
    
    final_df = pd.read_csv('data/UKB/fmri/descriptors_rewritten/fc_descriptors.csv')
    for idx, row in final_df.iterrows():
        final_df.at[idx, 'summary_rewritten'] = 'Compared with the cohort, ' + row['summary_rewritten']
    final_df.to_csv('data/UKB/fmri/descriptors_rewritten/fc_descriptors_.csv', index=False)
        
        