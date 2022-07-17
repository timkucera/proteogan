proteogan_config = {
    'batch_size': 128,
    'z_dim': 100,
    'label_dim': 50, # number of classes
    'seq_dim': 21, # length of amino acid alphabet + 1 (for padding)
    'seq_length': 2048,
    'kernel_size': 12,
    'strides': 8,
    'ac_weight': 135,
    'pad_sequence': True,
    'pad_labels': False,
}
