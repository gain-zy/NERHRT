# coding: utf-8
import torch
import logging

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
handler = logging.FileHandler("models/log.txt", encoding='UTF-8')
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

console = logging.StreamHandler()
console.setLevel(logging.DEBUG)

logger.addHandler(handler)
logger.addHandler(console)

logger.info("Start print log")


logger.info('='*100)
logger.info('*'*100)
logger.info('   23-5-4  最原始的中文版本')
logger.info('*'*100)
logger.info('-'*100)

USE_CUDA = torch.cuda.is_available()
device_nums = torch.cuda.device_count()
for i in range(device_nums):
    logger.info('This computer have gpu {}， the model is {} \n'.format(torch.device(i), torch.cuda.get_device_name(i)))
if USE_CUDA:
    device_tab = 1
    dev = "cuda:" + str(device_tab)
    cuda_0 = torch.cuda.get_device_name(0)
else:
    dev = "cpu"
device = torch.device(dev)

SEED = 50

batch_size = 30
embedding_size = 128
filter = 128
fusion_hidden = 256
hidden_size = 512
bert_hidden = 768
n_epochs = 80

op_matrix = None

hir_ele = 1e-5

learning_rate = 1e-3
weight_decay = 1e-5

beam_size = 5
gru_layers = 2
hir_layers = 100

# 以前这个平稳版本的学习率是：
#bert_learning_rate = 2e-5
#bert_weight_decay = 1e-4

# 改tree后学习率
bert_learning_rate = 5e-5
bert_weight_decay = 5e-5

target_ans_acc = 0.87

# conv
kernels = '3@5'

logger.info(' * bert_learning_rate是: {}; bert_weight_decay是: {} \n'.format(bert_learning_rate, bert_weight_decay))

get_trans_flag = True

PYLTP_DATA_DIR =r'/home/zy/research_v2/tool/Ltp_base2_v3_'

unk2word_vocab_path = './data/UNK2word_vocab'

pre_pairs_path = './data/pre_pairs.pkl'
processed_data_path = './data/processed.pkl'


basic_information = ' 总epoch -> {} ; 批次大小 -> {} ; 随机种子 -> {} ; 目标准确度 -> {} ; beam size : {}\n'
logger.info(basic_information.format(n_epochs, batch_size, SEED, target_ans_acc, beam_size))
basic_information = ' BERT的 学习率 -> {} ; BERT的 权重衰减 -> {} ; BERT 隐藏尺寸 -> {} \n'
logger.info(basic_information.format(bert_learning_rate, bert_weight_decay, bert_hidden))
basic_information = ' 其他 层 学习率 -> {} ; 其他 层 权重衰减 -> {} ; 其他层 隐藏尺寸 -> {} \n'
logger.info(basic_information.format(learning_rate, weight_decay, hidden_size))
#logger.info(' * 层次注意力是 hir_ele:  {} ;  层数是 : {}  \n'.format(hir_ele, hir_layers))


if "A5000" in cuda_0:
    logger.info(' * 使用cuda,' + cuda_0 + '这次运算在 ' + cuda_0)
    ltp_path = r'/home/zy/research_v2/tool/Ltp_base2_v3_'
    bert_path = r'/home/zy/research_v2/tool/have_fine_tune/chinese_roberta_wwm_ext_D/No/model'
elif 'Xp' in cuda_0:
    logger.info(' * 使用cuda,' + cuda_0 + '这次运算在 ' + cuda_0)
    ltp_path = r'/home/zy/research_v2/tools/Ltp_base2_v3_'
    bert_path = r'/home/zy/research_v2/tools/have_fine_tune/D_model/chinese_roberta_wwm_ext_D/model'
elif '3090' in cuda_0:
    logger.info(' * 使用cuda,' + cuda_0 + '这次运算在 ' + cuda_0)
    # ltp_path = r'/home/zy/research_v2/tools/Ltp_base2_v3_'
    # bert_path = r'/home/zy/research_v2/tools/have_fine_tune/D_model/chinese_roberta_wwm_ext_D/model'
    ltp_path = r'E:\research_v2\tools\Ltp_base2_v3_'
    bert_path = r'E:\research_v2\tools\have_fine_tune\chinese_roberta_wwm_ext_D\No\model'
elif 'A6000' in cuda_0:
    logger.info(' * 使用cuda,' + cuda_0 + '这次运算在 ' + cuda_0)
    ltp_path = r'/home/ccnu/zy/tools/Ltp_base2_v3_'
    bert_path = r'/home/ccnu/zy/tools/have_fine_tune/chinese_roberta_wwm_ext_D/No/model'
else:
    logger.info(' * 使用cuda,' + cuda_0 + '这次运算在 ' + cuda_0)
    assert 0==1
logger.info('查看路径：')
logger.info(' * bert_path: '+bert_path + '\n')
logger.info(' * ltp_path: '+ltp_path+'\n')

subgraph_type_list = ['number-word', 'number-number', 'word-word']

math23k_path = './data/Math22K.json'

