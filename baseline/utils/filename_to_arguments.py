import os
from collections import Counter

def process_relations_arguments(
    filename: str,
    annotations: list,
    components_boundaries: dict,
    components_types: dict
):
    """
    Processes relations from BRAT annotations, merging them into arguments when possible.

    Parameters:
        filename (str): Name of the file being processed.
        annotations (list): List of BRAT annotation lines split into components.
        components_boundaries (dict): Mapping from component IDs to their boundary spans.
        components_types (dict): Mapping from component IDs to their types.

    Notes:
        - Arguments are formed by merging related components.
        - Components are sorted based on their boundaries.
    """
    relations = []
    
    # Parse annotations and create relations
    for ann in annotations:
        if ann[0].startswith('R'):
            rel_info = ann[1]
            _, arg1, arg2 = rel_info.split(' ')
            arg1 = arg1.split(':')[1]
            arg2 = arg2.split(':')[1]

            relations.append({
                'id': ann[0],
                'args': [arg1, arg2],
                'args_boundaries': [components_boundaries[arg1], components_boundaries[arg2]],
                'args_types': [components_types[arg1], components_types[arg2]]
            })
    
    arguments = []
    used_relations = set()  # Track processed relations

    # Create a function to continuously merge relations that have common arguments
    def merge_relations():
        merged_relations = []
        available_relations = list(range(len(relations)))

        while available_relations:
            current_idx = available_relations.pop(0)
            if current_idx in used_relations:
                continue  # Skip already processed relations

            # Start with the current relation
            args = relations[current_idx]['args'][:]
            boundaries = relations[current_idx]['args_boundaries'][:]
            types = relations[current_idx]['args_types'][:]
            merged_relations_set = {relations[current_idx]['id']}

            # Try to merge with other relations until no more can be merged
            merged = True
            while merged:
                merged = False
                for idx in list(available_relations):
                    if idx in used_relations:
                        continue  # Skip already processed relations
                    # If any argument is common, merge them
                    if set(args) & set(relations[idx]['args']):
                        for arg, boundary, type_ in zip(relations[idx]['args'], relations[idx]['args_boundaries'], relations[idx]['args_types']):
                            if arg not in args:  # Add only unique arguments
                                args.append(arg)
                                boundaries.append(boundary)
                                types.append(type_)
                        merged_relations_set.add(relations[idx]['id'])
                        available_relations.remove(idx)  # Remove the merged relation
                        merged = True

            # After merging all possible relations, mark them as used
            used_relations.update(merged_relations_set)

            # Sort arguments by their boundary start values
            sorted_info = sorted(zip(args, boundaries, types), key=lambda x: x[1][0])

            # Unzip and remove inner lists
            sorted_args, sorted_boundaries, sorted_types = zip(*sorted_info)

            count_claims = len([1 for el in sorted_types if el in ['Claim', 'MajorClaim']])
            count_premises = len([1 for el in sorted_types if el in ['Premise']])

            if count_claims and count_premises:
                arguments.append({
                    'id': f'A{len(arguments)+1}',  # Generate a new argument ID
                    'args': list(sorted_args),
                    'args_boundaries': list(sorted_boundaries),
                    'args_types': list(sorted_types)
                })

    # Start merging relations
    merge_relations()

    return relations, arguments

def filename_to_arguments_number(folder: str, threshold: int = None):
    """
    Processes all files in the specified folder to count the number of arguments.

    Parameters:
        folder (str): Path to the folder containing BRAT files.
        threshold (int, optional): Minimum frequency of arguments for files to be included.

    Notes:
        - Counts argument occurrences per file and applies a threshold if specified.
        - Displays a summary of argument counts.
    """
    dataset = {}

    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith('.ann'):
                ann_path = os.path.join(root, file)
                txt_path = ann_path.replace('.ann', '.txt')
                
                if os.path.exists(txt_path):
                    filename = file.split('.')[0]
                    dataset[filename] = {}
                    with open(ann_path, 'r', encoding='utf-8') as ann_f, open(txt_path, 'r', encoding='utf-8') as txt_f:
                        annotations = [line.strip().split('\t') for line in ann_f]

                        components_boundaries = {ann[0]: (int(ann[1].split(' ')[1]), int(ann[1].split(' ')[2])) 
                                                 for ann in annotations if ann[0].startswith('T')}
                        components_types = {ann[0]: ann[1].split(' ')[0] 
                                            for ann in annotations if ann[0].startswith('T')}
                       
                        relations, arguments = process_relations_arguments(filename, annotations, components_boundaries, components_types)
                        
                        dataset[filename]['filename'] = filename
                        dataset[filename]['relations'] = relations
                        dataset[filename]['arguments'] = arguments
                        dataset[filename]['n'] = len(arguments)

    # Count occurrences of each argument count
    argument_counts = Counter(value['n'] for value in dataset.values())

    # Remove entries where the occurrence of their 'n' value is below the threshold
    if threshold is not None:
        print(f"Removing files whose argument count occurs less than {threshold} times.")
        dataset = {key: value for key, value in dataset.items() if argument_counts[value['n']] >= threshold}

    return dataset