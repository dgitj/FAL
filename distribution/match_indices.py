import pickle
import numpy as np
import os
from urllib.request import urlretrieve
import tarfile

# Class names for CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

def download_and_extract_cifar10():
    """Download and extract the CIFAR-10 dataset if not already available."""
    dataset_dir = 'cifar-10-batches-py'
    if not os.path.exists(dataset_dir):
        print("Downloading CIFAR-10 dataset...")
        url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        tarfile_path = 'cifar-10-python.tar.gz'
        urlretrieve(url, tarfile_path)
        
        print("Extracting files...")
        with tarfile.open(tarfile_path, 'r:gz') as tar:
            tar.extractall()
        os.remove(tarfile_path)
        print("Dataset extracted successfully.")
    return dataset_dir

def load_cifar10():
    """Load the complete CIFAR-10 dataset and return all labels."""
    dataset_dir = download_and_extract_cifar10()
    
    # Load training batches
    all_labels = []
    for batch_id in range(1, 6):
        batch_file = os.path.join(dataset_dir, f'data_batch_{batch_id}')
        with open(batch_file, 'rb') as f:
            batch_data = pickle.load(f, encoding='bytes')
            all_labels.extend(batch_data[b'labels'])
    
    # Load test batch
    test_file = os.path.join(dataset_dir, 'test_batch')
    with open(test_file, 'rb') as f:
        test_data = pickle.load(f, encoding='bytes')
        all_labels.extend(test_data[b'labels'])
    
    return np.array(all_labels)

def match_indices_to_labels(indices, all_labels):
    """Match indices to their labels and group by class."""
    results = {}
    
    # Group indices by their class
    for idx in indices:
        if idx < len(all_labels):
            label = int(all_labels[idx])
            class_name = class_names[label]
            
            if label not in results:
                results[label] = {
                    'class_name': class_name,
                    'count': 0,
                    'sample_indices': []
                }
            
            results[label]['count'] += 1
            # Store a few sample indices for reference (limit to 5)
            if len(results[label]['sample_indices']) < 5:
                results[label]['sample_indices'].append(idx)
        else:
            print(f"Warning: Index {idx} is out of bounds for dataset of size {len(all_labels)}")
    
    # Sort results by label
    sorted_results = {k: results[k] for k in sorted(results.keys())}
    return sorted_results

def main():
    # Example indices to match (replace with your list)
    sample_indices = [
  
             30321,
        35633,
        8249,
        2659,
        25920,
        38235,
        11887,
        17054,
        39883,
        17633,
        1432,
        9819,
        38233,
        30084,
        44666,
        12078,
        45297,
        18445,
        48419,
        10607,
        47689,
        39959,
        4651,
        42688,
        22238,
        39382,
        356,
        6317,
        14031,
        20507,
        7086,
        13700,
        11739,
        45801,
        47004,
        2673,
        39690,
        33998,
        42054,
        35695,
        22083,
        35424,
        29562,
        26591,
        35289,
        47528,
        36830,
        4961,
        47230,
        2117,
        20647,
        19182,
        30657,
        33552,
        18174,
        36387,
        17354,
        18278,
        21705,
        47229,
        32919,
        6184,
        37316,
        49874,
        21965,
        31356,
        19646,
        49361,
        16794,
        37221,
        15291,
        10253,
        31812,
        12293,
        44062,
        35123,
        16915,
        33178,
        7001,
        20993,
        7143,
        18820,
        44259,
        298,
        26429,
        31373,
        29191,
        4685,
        27873,
        26081,
        24420,
        22190,
        15086,
        401,
        15097,
        3694,
        15149,
        38011,
        2683,
        16075,
        5770,
        29653,
        11583,
        36252,
        14712,
        21204,
        14963,
        23482,
        41167,
        17165,
        41371,
        23804,
        20037,
        3070,
        32074,
        46123,
        31471,
        5682,
        4859,
        43264,
        20922,
        20902,
        17233,
        11061,
        13835,
        28438,
        15026,
        44379,
        39401,
        32703,
        37059,
        30763,
        11903,
        9423,
        16685,
        28740,
        31139,
        23207,
        32105,
        17303,
        40468,
        4081,
        21338,
        42151,
        8098,
        7866,
        17115,
        2233,
        637,
        5565,
        1726,
        44552,
        10715,
        16098,
        25792,
        21807,
        38299,
        33919,
        47328,
        23767,
        25023,
        2596,
        30383,
        47834,
        4668,
        14159,
        22875,
        32385,
        41983,
        13011,
        12393,
        46035,
        32985,
        6530,
        41965,
        23797,
        24852,
        47645,
        18605,
        10686,
        3008,
        32206,
        32999,
        2997,
        16160,
        14231,
        49129,
        23238,
        49930,
        27660,
        32025,
        47038,
        41600,
        24389,
        21359,
        44299,
        15536,
        41324,
        37046,
        42931,
        18924,
        48512,
        32218,
        35136,
        14137,
        710,
        30895,
        49890,
        7472,
        33737,
        30906,
        33309,
        9530,
        47905,
        2159,
        42671,
        5057,
        8258,
        11310,
        1885,
        4322,
        8363,
        234,
        3636,
        49418,
        3443,
        37621,
        37699,
        35690,
        29572,
        9594,
        30487,
        3880,
        14514,
        42041,
        26504,
        31837,
        7031,
        11098,
        32993,
        18851,
        39227,
        9031,
        28467,
        28223,
        19491,
        35951,
        41553,
        2427,
        9709
    ]
    
    print("Loading CIFAR-10 dataset...")
    all_labels = load_cifar10()
    print(f"Dataset loaded, total size: {len(all_labels)} images")
    
    # Match the indices to labels
    results = match_indices_to_labels(sample_indices, all_labels)
    
    # Print summary
    print("\nLabel Distribution Summary:")
    print("==========================")
    for label, data in results.items():
        print(f"Class {label} ({data['class_name']}): {data['count']} images")
    
    # Print detailed breakdown
    print("\nDetailed Class Breakdown:")
    print("=======================")
    for label, data in results.items():
        print(f"\nClass {label}: {data['class_name']}")
        print(f"Count: {data['count']} images")
        print(f"Sample indices: {data['sample_indices']}")

if __name__ == "__main__":
    main()