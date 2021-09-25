import sys, os, time
from data.util import load
from metrics.similarity import mmd
from metrics.conditional import mrr

PATH = os.path.dirname(os.path.realpath(__file__))

model = sys.argv[1]
split = int(sys.argv[2])

assert model in ['proteogan']
assert split in [1,2,3,4,5]

train = load('train',s plit)
test = load('test', split)
val = load('val', split)

if model == 'proteogan':
    from models.proteogan import Trainer
    model_config = {
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

trainer = Trainer(
    batch_size = model_config['batch_size'],
    plot_every = 10,
    checkpoint_every = 10,
    train_data = train,
    test_data = test,
    val_data = val,
    path = '{}/split_{}/{}'.format(PATH,split,model),
    config = model_config,
    pad_sequence = model_config['pad_sequence'],
    pad_labels = model_config['pad_labels'],
    short_val = model == 'transformer',
)

trainer.train(epochs=300, progress=True, restore=False)
