import sys
import tensorflow as tf

sys.path.append(".")
from os.path import join
from src.utils.configs import cfg
from src.embedding.concept.evaluation import ConceptEvaluation as Evaluator
from src.embedding.concept.dataset import ConceptDataset as CDataset
from src.embedding.concept.model import ConceptModel as Model
from src.utils.graph_handler import GraphHandler
from src.utils.record_log import RecordLog
from src.embedding.mortality.data_transformer import tesan_trans, med2vec_trans, mce_trans, glove_trans
import warnings

warnings.filterwarnings('ignore')

logging = RecordLog()

if __name__ == '__main__':
    tf.compat.v1.disable_v2_behavior()
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=None, allow_growth=True)
    graph_config = tf.compat.v1.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    data_set = CDataset()
    data_set.prepare_data(cfg.visit_threshold)
    data_set.build_dictionary()
    data_set.load_data()
    sess = tf.compat.v1.Session(config=graph_config)
    with tf.compat.v1.variable_scope('concept_embedding') as scope:
        model = Model(scope.name, data_set)
        model.final_weights = tf.convert_to_tensor(tesan_trans("cbow"), dtype=tf.float32)

    graph_handler = GraphHandler(model, logging)
    graph_handler.initialize(sess)
    evaluator = Evaluator(model, logging)
    icd_nns = evaluator.get_nns_p_at_top_k(sess, 'ICD')
    icd_weigh_scores = evaluator.get_clustering_nmi(sess, 'ICD')
    ccs_nns = evaluator.get_nns_p_at_top_k(sess, 'CCS')
    ccs_weigh_scores = evaluator.get_clustering_nmi(sess, 'CCS')

    print('validating the embedding performance .....')
    log_str = "weight: %s %s %s %s" % (icd_weigh_scores, ccs_weigh_scores, icd_nns, ccs_nns)
    logging.add(log_str)