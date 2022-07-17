import sys, os, time, tempfile
from data.util import load
from metrics.similarity import mmd
from metrics.conditional import mrr

PATH = os.path.dirname(os.path.realpath(__file__))

model = sys.argv[1]
split = int(sys.argv[2])

assert model in ['proteogan']
assert split in [1,2,3,4,5]

train = load('train', split)
test = load('test', split)
val = load('val', split)

if model == 'proteogan':
    from models.proteogan import Trainer
    from models.config import proteogan_config as model_config

print('Reloading')
ckpt_path = '{}/split_{}/{}'.format(PATH,split,model)

with tempfile.TemporaryDirectory() as tmp_dir:
    trainer = Trainer(
        batch_size = model_config['batch_size'],
        path = '{}/split_{}/{}'.format(PATH,split,model),
        config = model_config,
        pad_sequence = model_config['pad_sequence'],
        pad_labels = model_config['pad_labels'],
    )

    trainer.restore(path=ckpt_path, best=True)

    generated = trainer.generate(labels=test['labels'], progress=True, seed=0)
    metrics = {
        'MMD': mmd(generated, test['sequence']),
        'MRR': mrr(generated, test['labels'], test['sequence'], test['labels']),
    }
    print(metrics)
