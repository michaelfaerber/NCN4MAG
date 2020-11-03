from model import *
from training import *
from data import *
import pickle
from datetime import datetime

random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# set up training
data = get_bucketized_iterators("/pfs/work7/workspace/scratch/ucgvm-input-0/input/",batch_size = 64,len_context_vocab = 20000,len_title_vocab = 20000,len_aut_vocab = 20000)
PAD_IDX = data.ttl.vocab.stoi['<pad>']
cntxt_vocab_len = len(data.cntxt.vocab)
aut_vocab_len = len(data.aut.vocab)
ttl_vocab_len = len(data.ttl.vocab)



net = NeuralCitationNetwork(context_filters=[4,4,5,6,7],author_filters=[1,2],context_vocab_size=cntxt_vocab_len,title_vocab_size=ttl_vocab_len,author_vocab_size=aut_vocab_len,pad_idx=PAD_IDX,num_filters=256,
                            embed_size=128,
                            authors=False,
                            num_layers=1,
                            hidden_size=256,
                            dropout_p=0.2,
                            show_attention=True)
net.to(DEVICE)

train_losses, valid_losses = train_model(model = net,train_iterator = data.train_iter,valid_iterator = data.valid_iter,lr = 0.001,pad = PAD_IDX, model_name = "model_"+(datetime.today().strftime('%d_%m_%Y')))

with open('train_losses', 'wb') as fp:
	pickle.dump(train_losses, fp)

with open('valid_losses', 'wb') as fp:
	pickle.dump(valid_losses, fp)