import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
import pandas as pd
import models.sentence_select.utils as utils
from models.utils import load_jsonl_gz
# Transformer PL
from models.transformers.model import Transformer_PL
import os
import numpy as np
import pathlib
import spacy
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)
nlp = spacy.load('en_core_web_sm', disable=['tokenizer', 'ner', 'textcat'])


class BaseModel(object):
    """
    Abstract Class for DecSum task
    """
    def __init__(self, args):
        self.args = args
        self._load_model(args)
        self.cache_dir = os.path.join(
            self.args.data_dir, "cache_result", f"{self.args.data}",
            f"{self.args.num_review}reviews", self.args.data_split,
            self.args.model)
        pathlib.Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Cache directory: {self.cache_dir}")

    def _get_cache_name(self, cache):
        if self.args.debug:
            return f"{cache}_debug"
        return cache

    def _load_model(self, args):
        "load model name and model"
        raise NotImplementedError

    def _get_data(self):
        if self.args.data == "yelp":
            logger.info("Loading data")
            test_list = pd.DataFrame(load_jsonl_gz(f"{self.args.data_dir}/{self.args.num_review}reviews/{self.args.data_split}.jsonl.gz"))
            logger.info("Segmenting data")
            X_test_notes, y_test, business = [], [], []
            for index, row in test_list.iterrows():
                sentences = utils.segment_sentence(self.args.segment, row)
                X_test_notes.extend(sentences)
                y_test.extend([row['avg_score']]*len(sentences))
                business.extend([row['business'] for _ in range(len(sentences))])
                if self.args.debug:
                    break
            return X_test_notes, business, y_test
        else:
            raise(f"provided dataset ({self.args.data}) is invalid.")

    def _process_cache_file(self, cache, inputs, func, labels=None):
        cache = self._get_cache_name(cache)
        # currently only cache match methods
        cached_filepath = os.path.join(self.cache_dir, f"{cache}.npy")
        if os.path.exists(cached_filepath):
            logger.info(f"Load prediction cache at {cached_filepath}")
            return np.load(cached_filepath, allow_pickle=True)
        else:
            logger.info(f"No prediction cache. Run prediction")
            preds = func(inputs, labels=labels)
            preds = np.array(preds)
            np.save(cached_filepath, preds)
            return preds

    def run_model(self, inputs, labels=None, cache=False):
        # check cache file and return probability of each input, only for all and decsum now
        if cache:
            return self._process_cache_file(cache, inputs, self._run_model, labels=labels)
        else:
            return self._run_model(inputs, labels)

    def _run_model(self, inputs, labels=None):
        "return probability of each input for a specific model"
        raise NotImplementedError

    def predict(self):
        segments, businesses, avg_scores = self._get_data()  # segmented review
        segment_preds = self.run_model(
            segments, avg_scores,
            cache=f"segment_predicted_score" if "DecSum" in self.args.segment else None
        )
        
        if "DecSum" not in self.args.segment:
            if self.args.data!="yelp":
                raise(f"provided dataset ({self.args.data}) is invalid.")
            utils.find_highest_score(segment_preds, businesses, segments, avg_scores, self.args, top_k=self.args.return_candidates)

        else:
            logger.info("Run DecSum")
            """
            Get DecSum hyperparameters
            """
            K = self.args.num_sentences
            alpha = self.args.alpha
            beta = self.args.beta
            gamma = self.args.gamma
            if alpha == 0 and beta == 0 and gamma == 0:
                logger.error("Invalid hyperparameters. You can't set all three parameters to zero.")
                import sys
                sys.exit()
            """
            Get predicted scores with all reviews for computing decision faithfulness
            """
            tmp = self.args.segment
            self.args.segment = "all"
            all_reviews, tmp_businesses, tmp_avg_scores = self._get_data() # segmented note
            predicted_all_avg_score = self.run_model(all_reviews,
                                                     cache="all_review_predicted_score")
            all_df = pd.DataFrame({"business": tmp_businesses, "all_predicted_score": predicted_all_avg_score})

            df = pd.DataFrame({"business": businesses,
                               "segment_predicted_score": segment_preds})
            df = pd.merge(df, all_df, on=["business"])
            self.args.segment = tmp
            # we want scores of selected chunks as close as predicted scores using all reviews
            if self.args.data == "yelp":
                df['segment'] = segments
            else:
                raise(f"provided dataset ({self.args.data}) is invalid.")
                df['input'] = segments[0]
                df['query'] = segments[1]
            """
             Compute decsion faithfulness score for the First run. Add epsilon to prevent nan.
            """
            df['decsum_scores'] = -np.log(abs(segment_preds - df["all_predicted_score"].values) + 1e-6)
            decsum_output = []
            decsum_output_sent_idx = []

            for b in tmp_businesses:
                df_b = df[df['business'] == b]
                if self.args.data == "yelp":
                    ret, sent_idx = utils.run_decsum(
                        df_b['segment'].values, b, df_b['segment_predicted_score'].values, 
                        df_b['decsum_scores'].values, df_b['all_predicted_score'].values[0],
                        K, alpha, beta, gamma, self,
                        query=None, beam_size=4,
                    )
                else:
                    raise(f"provided dataset ({self.args.data}) is invalid.")
                decsum_output.extend(ret)
                decsum_output_sent_idx.extend(sent_idx)
                if self.args.debug:
                    logger.info(f"Stop early for debugging")
                    break
            """
            Run on selected sentences
            """
            if self.args.data == "yelp":
                predicted_avg_score = self.run_model(decsum_output)
            else:
                raise(f"provided dataset ({self.args.data}) is invalid.")
            # print(len(predicted_avg_score))
            # print(len(decsum_output))
            # print(len(tmp_avg_scores))
            # print(len(decsum_output_sent_idx))
            utils.find_highest_score(predicted_avg_score, tmp_businesses, decsum_output, tmp_avg_scores, self.args, sent_idx=decsum_output_sent_idx)


class Transformer(BaseModel):
    def __init__(self, args):
        super(Transformer, self).__init__(args)

    def _load_model(self, args):
        logger.info("loading model")
        if args.device == 'cpu':
            self.device = torch.device(f"cpu")
        else:
            self.device = torch.device(f"cuda:{args.device}")
        
        if args.num_review == 50:
            checkpoint_path = args.trained_model_path
        else:
            raise("Invalid number of reviews. Can't load Longformer.")
        self.model = Transformer_PL.load_from_checkpoint(
            checkpoint_path=checkpoint_path
        )
        if args.device != 'cpu':
            self.model.half()
        self.model.freeze()
        self.model.eval()
        self.model.to(self.device)
        model_name = '.chkpt'

        if args.feature_used == "all":
            model_name = "feature_text_" + model_name
        elif args.feature_used == "all_but_notes":
            model_name = "feature_" + model_name
        else:
            model_name = "text_" + model_name

        self.model_name = model_name
        self.args = args

    def _run_model(self, inputs, labels=None):
        if not labels:
            labels = [0]*len(inputs)
        dataloader = self._buildDataLoader(inputs, labels)
        return self._run(dataloader)

    def _buildDataLoader(self, notes, labels):
        data_size = len(notes)
        batch_size = 32
        n_batch = data_size // batch_size + 1
        for i in range(n_batch):
            batch_notes = notes[i * batch_size: (i + 1) * batch_size]
            batch_labels = labels[i * batch_size: (i + 1) * batch_size]
            #B = len(batch_notes)
            if len(batch_notes) == 0:
                break
            encodings = self.model.tokenizer.batch_encode_plus(
                batch_notes,
                truncation=True,
                padding=True,
                max_length=self.model.hparams.max_seq_length,
                return_tensors='pt'
            ) #[B, max_len in this batch]
            
            batch_labels = torch.FloatTensor(batch_labels).view(-1, 1)

            yield encodings['input_ids'], encodings['attention_mask'], batch_labels

    def _run(self, dataloader):
        preds = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                input_ids, attention_mask, labels = map(lambda x: x.to(self.device), batch)
                inputs = {"input_ids": input_ids,
                          "attention_mask": attention_mask}
                pred = self.model(**inputs)[0] #logits in classification/ score in regression
                preds.extend([p.item() for p in pred])
        return preds