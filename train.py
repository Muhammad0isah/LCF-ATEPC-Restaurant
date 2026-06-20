#test
import argparse
import json
import logging
import os, sys
import random

import seqeval.metrics
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
from time import strftime, localtime
import numpy as np
import torch
import torch.nn.functional as F
try:
    from transformers.optimization import AdamW
except ImportError:
    from torch.optim import AdamW
from transformers.models.bert.modeling_bert import BertModel
from transformers import BertTokenizer
from seqeval.metrics import classification_report
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from utils.data_utils import ATEPCProcessor, convert_examples_to_features
from model.lcf_atepce import LCF_ATEPC

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
os.makedirs('logs', exist_ok=True)
time = '{}'.format(strftime("%y%m%d-%H%M%S", localtime()))
log_file = 'logs/{}.log'.format(time)
logger.addHandler(logging.FileHandler(log_file))
logger.info('log file: {}'.format(log_file))



def main(config):
    args = config

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    processor = ATEPCProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list) + 1

    datasets = {
        'restaurant': "atepc_datasets/restaurant",
    }
    pretrained_bert_models = {
        'restaurant': "bert-base-uncased",
    }

    args.bert_model = pretrained_bert_models[args.dataset]
    args.data_dir = datasets[args.dataset]

    def convert_polarity(examples):
        for i in range(len(examples)):
            polarities = []
            for polarity in examples[i].polarity:
                if polarity == 2:
                    polarities.append(1)
                else:
                    polarities.append(polarity)
            examples[i].polarity = polarities

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
    train_examples = processor.get_train_examples(args.data_dir)
    eval_examples = processor.get_test_examples(args.data_dir)
    num_train_optimization_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    bert_base_model = BertModel.from_pretrained(args.bert_model)
    bert_base_model.config.num_labels = num_labels

    if args.dataset in {'camera', 'car', 'phone', 'notebook'}:
        convert_polarity(train_examples)
        convert_polarity(eval_examples)
        model = LCF_ATEPC(bert_base_model, args=args)
    else:
        model = LCF_ATEPC(bert_base_model, args=args)

    for arg in vars(args):
        logger.info('>>> {0}: {1}'.format(arg, getattr(args, arg)))

    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.00001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.00001}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, weight_decay=0.00001)
    eval_features = convert_examples_to_features(eval_examples, label_list, args.max_seq_length,
                                                 tokenizer)
    all_spc_input_ids = torch.tensor([f.input_ids_spc for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    all_polarities = torch.tensor([f.polarities for f in eval_features], dtype=torch.long)
    all_emotions = torch.tensor([f.emotions for f in eval_features], dtype=torch.long)
    all_valid_ids = torch.tensor([f.valid_ids for f in eval_features], dtype=torch.long)
    all_lmask_ids = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_spc_input_ids, all_input_mask, all_segment_ids, all_label_ids,all_polarities,all_valid_ids, all_lmask_ids,all_emotions)
    eval_sampler = RandomSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    def evaluate(eval_ATE=True, eval_APC=True,eval_emotion=True):
        def build_per_class_accuracy(y_true, y_pred, label_names):
            rows = []
            for label_id, label_name in label_names.items():
                label_mask = y_true == label_id
                support = int(label_mask.sum().item())
                correct = int(((y_pred == label_id) & label_mask).sum().item())
                accuracy = round((correct / support) * 100, 2) if support else 0.0
                rows.append({
                    'label': label_name,
                    'support': support,
                    'correct': correct,
                    'accuracy': accuracy
                })
            return rows

        def format_per_class_accuracy(rows):
            return '; '.join(
                f"{row['label']}: n={row['support']}, correct={row['correct']}, acc={row['accuracy']}%"
                for row in rows
            )

        def build_token_metrics(y_true_seq, y_pred_seq, label_names):
            true_flat = [label for sentence in y_true_seq for label in sentence]
            pred_flat = [label for sentence in y_pred_seq for label in sentence]
            precision, recall, f1, support = precision_recall_fscore_support(
                true_flat,
                pred_flat,
                labels=label_names,
                zero_division=0
            )
            rows = []
            for i, label_name in enumerate(label_names):
                rows.append({
                    'label': label_name,
                    'precision': round(float(precision[i]), 4),
                    'recall': round(float(recall[i]), 4),
                    'f1': round(float(f1[i]), 4),
                    'support': int(support[i])
                })
            return rows

        def format_token_metrics(rows):
            return '; '.join(
                f"{row['label']}: precision={row['precision']}, recall={row['recall']}, "
                f"f1={row['f1']}, support={row['support']}"
                for row in rows
            )

        apc_result = {'max_apc_test_acc': 0, 'max_apc_test_f1': 0}
        ate_result = {'max_ate_test_f1': 0, 'per_tag_metrics': [], 'per_tag_metrics_text': ''}
        emotion_result = {'max_emotion_test_acc': 0, 'max_emotion_test_f1': 0}
        y_true = []
        y_pred = []
        apc_test_correct, apc_test_total = 0, 0
        emotion_test_correct, emotion_test_total = 0, 0
        test_apc_logits_all, test_polarities_all = None, None
        test_emotion_logits_all, test_emotions_all = None, None
        model.eval()
        label_map = {i: label for i, label in enumerate(label_list, 1)}
        polarity_label_names = {0: 'Neutral', 1: 'Negative', 2: 'Positive'}
        emotion_label_names = {0: 'Anger-merged', 1: 'Surprise', 2: 'Joy'}
        for input_ids_spc, input_mask, segment_ids, label_ids, polarities, valid_ids, l_mask,emotions in eval_dataloader:
            input_ids_spc = input_ids_spc.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            valid_ids = valid_ids.to(device)
            label_ids = label_ids.to(device)
            polarities = polarities.to(device)
            l_mask = l_mask.to(device)
            emotions = emotions.to(device)
            with torch.no_grad():
                # ate_logits, apc_logits = 
                ate_logits, apc_logits,emotion_logits = model(input_ids_spc, segment_ids, input_mask,
                                               valid_ids=valid_ids, polarities=polarities, attention_mask_label=l_mask,emotions=emotions)
            if eval_APC:
                polarities = model.get_batch_polarities(polarities)
                apc_test_correct += (torch.argmax(apc_logits, -1) == polarities).sum().item()
                apc_test_total += len(polarities)
                if test_polarities_all is None:
                    test_polarities_all = polarities
                    test_apc_logits_all = apc_logits
                else:
                    test_polarities_all = torch.cat((test_polarities_all, polarities), dim=0)
                    test_apc_logits_all = torch.cat((test_apc_logits_all, apc_logits), dim=0)

            if eval_emotion:
                emotions = model.get_batch_emotions(emotions)
                emotion_test_correct += (torch.argmax(emotion_logits, -1) == emotions).sum().item()
                emotion_test_total += len(emotions)

                if test_emotions_all is None:
                    test_emotions_all = emotions
                    test_emotion_logits_all = emotion_logits
                else:
                    test_emotions_all = torch.cat((test_emotions_all, emotions), dim=0)
                    test_emotion_logits_all = torch.cat((test_emotion_logits_all, emotion_logits), dim=0)

            if eval_ATE:
                if not args.use_bert_spc:
                    label_ids = model.get_batch_token_labels_bert_base_indices(label_ids)
                ate_logits = torch.argmax(F.log_softmax(ate_logits, dim=2), dim=2)
                ate_logits = ate_logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()
                input_mask = input_mask.to('cpu').numpy()
                for i, label in enumerate(label_ids):
                    temp_1 = []
                    temp_2 = []
                    for j, m in enumerate(label):
                        if j == 0:
                            continue
                        elif label_ids[i][j] == len(label_list):
                            y_true.append(temp_1)
                            y_pred.append(temp_2)
                            break
                        else:
                            temp_1.append(label_map.get(label_ids[i][j], 'O'))
                            temp_2.append(label_map.get(ate_logits[i][j], 'O'))

        if eval_APC:
            apc_preds = torch.argmax(test_apc_logits_all, -1).cpu()
            apc_true = test_polarities_all.cpu()
            test_acc = apc_test_correct / apc_test_total
            test_f1 = f1_score(apc_true, apc_preds, labels=[0, 1, 2], average='macro')
            apc_per_class = build_per_class_accuracy(apc_true, apc_preds, polarity_label_names)
            test_acc = round(test_acc * 100, 2)
            test_f1 = round(test_f1 * 100, 2)
            apc_result = {
                'max_apc_test_acc': test_acc,
                'max_apc_test_f1': test_f1,
                'per_class_accuracy': apc_per_class,
                'per_class_accuracy_text': format_per_class_accuracy(apc_per_class)
            }

        if eval_ATE:
            # report = classification_report(y_true, y_pred, digits=4)
            # print(report)
            # print("y_true",y_true)
            # print("*"*50)
            # print("y_pred",y_pred)
            ate_f1 = seqeval.metrics.f1_score(y_true, y_pred, average='macro')
            ate_per_tag = build_token_metrics(y_true, y_pred, ['B-ASP', 'I-ASP', 'O'])

            # tmps = report.split()
            # ate_result = round(float(tmps[7]) * 100, 2)
            ate_result = {
                'max_ate_test_f1': round(float(ate_f1) * 100, 2),
                'per_tag_metrics': ate_per_tag,
                'per_tag_metrics_text': format_token_metrics(ate_per_tag)
            }

        if eval_emotion:
            emotion_preds = torch.argmax(test_emotion_logits_all, -1).cpu()
            emotion_true = test_emotions_all.cpu()
            emotion_f1 = f1_score(emotion_true, emotion_preds, labels=[0, 1, 2], average='macro')
            emotion_acc = accuracy_score(emotion_true, emotion_preds)
            emotion_per_class = build_per_class_accuracy(emotion_true, emotion_preds, emotion_label_names)
            emotion_acc = round(float(emotion_acc) * 100, 2)
            emotion_f1 = round(float(emotion_f1) * 100, 2)
            emotion_result = {
                'max_emotion_test_acc': emotion_acc,
                'max_emotion_test_f1': emotion_f1,
                'per_class_accuracy': emotion_per_class,
                'per_class_accuracy_text': format_per_class_accuracy(emotion_per_class)
            }
        return apc_result, ate_result,emotion_result
    def save_model(path):
        # Save a trained model and the associated configuration,
        # Take care of the storage!
        os.makedirs(path, exist_ok=True)
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        model_to_save.save_pretrained(path)
        tokenizer.save_pretrained(path)
        label_map = {i: label for i, label in enumerate(label_list, 1)}
        model_config = {"bert_model": args.bert_model, "do_lower": True, "max_seq_length": args.max_seq_length,
                        "num_labels": len(label_list) + 1, "label_map": label_map}
        json.dump(model_config, open(os.path.join(path, "config.json"), "w"))
        logger.info('save model to: {}'.format(path))
    def train():
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_spc_input_ids = torch.tensor([f.input_ids_spc for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in train_features], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in train_features], dtype=torch.long)
        all_polarities = torch.tensor([f.polarities for f in train_features], dtype=torch.long)
        all_emotions = torch.tensor([f.emotions for f in train_features], dtype=torch.long)

        train_data = TensorDataset(all_spc_input_ids, all_input_mask, all_segment_ids,
                                   all_label_ids, all_polarities,all_valid_ids, all_lmask_ids,all_emotions)
        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        max_apc_test_acc = 0
        max_apc_test_f1 = 0
        max_ate_test_f1 = 0
        max_emotion_test_acc = 0
        max_emotion_test_f1 = 0
        best_apc_per_class_accuracy_text = ''
        best_emotion_per_class_accuracy_text = ''
        best_ate_per_tag_metrics_text = ''
        global_step = 0
        for epoch in range(int(args.num_train_epochs)):
            logger.info('#' * 80)
            logger.info('Train {} Epoch{}'.format(args.seed, epoch + 1, args.data_dir))
            logger.info('#' * 80)
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(train_dataloader):
                model.train()
                batch = tuple(t.to(device) for t in batch)
                input_ids_spc, input_mask, segment_ids, label_ids, polarities,valid_ids, l_mask,emotions = batch
                loss_ate, loss_apc,loss_emo = model(input_ids_spc,segment_ids,input_mask, label_ids, polarities, valid_ids,
                                           l_mask,emotions)

                # Calculate the weighted loss
                loss =  loss_ate + loss_apc + loss_emo
                loss.backward()
                nb_tr_examples += input_ids_spc.size(0)
                nb_tr_steps += 1
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % args.eval_steps == 0:
                    if epoch >= args.num_train_epochs - 2 or args.num_train_epochs <= 2:
                        # evaluate in last 2 epochs
                        apc_result, ate_result,emotion_result = evaluate(eval_ATE=not args.use_bert_spc)
                        # apc_result, ate_result = evaluate()
                        # path = '{0}/{1}_{2}_apcacc_{3}_apcf1_{4}_atef1_{5}'.format(
                        #     args.output_dir,
                        #     args.dataset,
                        #     args.local_context_focus,
                        #     round(apc_result['max_apc_test_acc'], 2),
                        #     round(apc_result['max_apc_test_f1'], 2),
                        #     round(ate_result, 2)
                        # )
                        # if apc_result['max_apc_test_acc'] > max_apc_test_acc or \
                        #     apc_result['max_apc_test_f1'] > max_apc_test_f1 or \
                        #     ate_result > max_ate_test_f1:
                        #     save_model(path)

                        if apc_result['max_apc_test_acc'] > max_apc_test_acc:
                            max_apc_test_acc = apc_result['max_apc_test_acc']
                            best_apc_per_class_accuracy_text = apc_result.get('per_class_accuracy_text', '')
                        if apc_result['max_apc_test_f1'] > max_apc_test_f1:
                            max_apc_test_f1 = apc_result['max_apc_test_f1']
                        if ate_result['max_ate_test_f1'] > max_ate_test_f1:
                            max_ate_test_f1 = ate_result['max_ate_test_f1']
                            best_ate_per_tag_metrics_text = ate_result.get('per_tag_metrics_text', '')
                        if emotion_result['max_emotion_test_acc'] > max_emotion_test_acc:
                            max_emotion_test_acc = emotion_result['max_emotion_test_acc']
                            best_emotion_per_class_accuracy_text = emotion_result.get('per_class_accuracy_text', '')
                        if emotion_result['max_emotion_test_f1'] > max_emotion_test_f1:
                            max_emotion_test_f1 = emotion_result['max_emotion_test_f1']

                        current_apc_test_acc = apc_result['max_apc_test_acc']
                        current_apc_test_f1 = apc_result['max_apc_test_f1']
                        current_ate_test_f1 = round(ate_result['max_ate_test_f1'], 2)
                        current_emotion_test_acc = emotion_result['max_emotion_test_acc']
                        current_emotion_test_f1 = emotion_result['max_emotion_test_f1']

                        logger.info('*' * 80)
                        logger.info('Train {} Epoch{}, Evaluate for {}'.format(args.seed, epoch + 1, args.data_dir))
                        logger.info(f'APC_test_acc: {current_apc_test_acc}(max: {max_apc_test_acc})  '
                                    f'APC_test_f1: {current_apc_test_f1}(max: {max_apc_test_f1})')
                        if args.use_bert_spc:
                            logger.info(f'ATE_test_F1: {current_apc_test_f1}(max: {max_apc_test_f1})'
                                        f' (Unreliable since `use_bert_spc` is "True".)')
                        else:
                            logger.info(f'ATE_test_f1: {current_ate_test_f1}(max:{max_ate_test_f1})')
                        logger.info(
                            f'Emotion_test_acc: {current_emotion_test_acc}(max: {max_emotion_test_acc})'
                            f' Emotion_test_f1: {current_emotion_test_f1}(max: {max_emotion_test_f1})')
                        if 'per_class_accuracy_text' in apc_result:
                            logger.info(f"APC_per_class_accuracy: {apc_result['per_class_accuracy_text']}")
                        if 'per_class_accuracy_text' in emotion_result:
                            logger.info(f"Emotion_per_class_accuracy: {emotion_result['per_class_accuracy_text']}")
                        if ate_result.get('per_tag_metrics_text'):
                            logger.info(f"ATE_per_tag_metrics: {ate_result['per_tag_metrics_text']}")
                        logger.info('*' * 80)

        logger.info(f'BEST_APC_per_class_accuracy: {best_apc_per_class_accuracy_text}')
        logger.info(f'BEST_Emotion_per_class_accuracy: {best_emotion_per_class_accuracy_text}')
        logger.info(f'BEST_ATE_per_tag_metrics: {best_ate_per_tag_metrics_text}')
        return [max_apc_test_acc, max_apc_test_f1, max_ate_test_f1,max_emotion_test_acc,max_emotion_test_f1]

    return train()

def parse_experiments(path):
    configs = []
    opt = argparse.ArgumentParser()
    with open(path, "r", encoding='utf-8') as reader:
        json_config = json.loads(reader.read())
    for id, config in json_config.items():
        # Hyper Parameters
        parser = argparse.ArgumentParser()
        parser.add_argument("--dataset", default=config['dataset'], type=str)
        parser.add_argument("--output_dir", default=config['output_dir'], type=str)
        parser.add_argument("--SRD", default=int(config['SRD']), type=int)
        parser.add_argument("--learning_rate", default=float(config['learning_rate']), type=float,
                            help="The initial learning rate for Adam.")
        parser.add_argument("--use_unique_bert", default=bool(config['use_unique_bert']), type=bool)
        parser.add_argument("--use_bert_spc", default=bool(config['use_bert_spc_for_apc']), type=bool)
        parser.add_argument("--local_context_focus", default=config['local_context_focus'], type=str)
        parser.add_argument("--num_train_epochs", default=float(config['num_train_epochs']), type=float,
                            help="Total number of training epochs to perform.")
        parser.add_argument("--train_batch_size", default=int(config['train_batch_size']), type=int,
                            help="Total batch size for training.")
        parser.add_argument("--dropout", default=float(config['dropout']), type=int)
        parser.add_argument("--num_emotion_labels", default=int(config.get('num_emotion_labels', 3)), type=int)
        parser.add_argument("--max_seq_length", default=int(config['max_seq_length']), type=int)
        parser.add_argument("--eval_batch_size", default=32, type=int, help="Total batch size for eval.")
        parser.add_argument("--eval_steps", default=20, help="evaluate per steps")
        parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                            help="Number of updates steps to accumulate before performing a backward/update pass.")
        parser.add_argument("--config_path", default='experiments.json', type=str,
                            help='Path of experiments config file')
        configs.append(parser.parse_args())
    return configs
if __name__ == "__main__":
    experiments = argparse.ArgumentParser()
    experiments.add_argument('--config_path', default='experiments.json', type=str,
                             help='Path of experiments config file')
    experiments = experiments.parse_args()
    # from utils.Pytorch_GPUManager import GPUManager
    # index = GPUManager().auto_choice()
    # device = torch.device("cuda" + str(index) if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_configs = parse_experiments(experiments.config_path)
    n = 5
    for config in exp_configs:
        logger.info('-' * 80)
        logger.info('Config {} (totally {} configs)'.format(exp_configs.index(config) + 1, len(exp_configs)))
        results = []
        max_apc_test_acc, max_apc_test_f1, max_ate_test_f1, max_emotion_test_acc, max_emotion_test_f1 = 0, 0, 0, 0, 0
        for i in range(n):
            config.device = device
            config.seed = i + 1
            logger.info('No.{} training process of {}'.format(i + 1, n))
            # Assume that main(config) now returns emotion_test_acc and emotion_test_f1 as well
            apc_test_acc, apc_test_f1, ate_test_f1, emotion_test_acc, emotion_test_f1 = main(config)
            if apc_test_acc > max_apc_test_acc:
                max_apc_test_acc = apc_test_acc
            if apc_test_f1 > max_apc_test_f1:
                max_apc_test_f1 = apc_test_f1
            if ate_test_f1 > max_ate_test_f1:
                max_ate_test_f1 = ate_test_f1
            if emotion_test_acc > max_emotion_test_acc:
                max_emotion_test_acc = emotion_test_acc
            if emotion_test_f1 > max_emotion_test_f1:
                max_emotion_test_f1 = emotion_test_f1
            logger.info(
                'max_ate_test_f1:{} max_apc_test_acc: {}\tmax_apc_test_f1: {} \tmax_emotion_test_acc: {}\tmax_emotion_test_f1: {}'
                .format(max_ate_test_f1, max_apc_test_acc, max_apc_test_f1, max_emotion_test_acc, max_emotion_test_f1))
