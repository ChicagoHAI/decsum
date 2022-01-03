import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
import numpy as np
import spacy
from scipy.stats import wasserstein_distance
from sklearn.metrics import f1_score, mean_squared_error
import os
import pandas as pd
from summa import summarizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)
nlp = spacy.load('en_core_web_sm', disable=['tokenizer', 'ner', 'tagger','textcat'])


class SentBert:
    """Singleton sentenceBert class"""
    __model = None

    def __new__(self, model_name='distilbert-base-nli-stsb-mean-tokens', device='cuda'):
        if not SentBert.__model:
            SentBert.__model = SentenceTransformer(model_name, device=device)
        return SentBert.__model


def segment_sentence(segment, notes, datapath=None, stay=None):
    # note = note.replace("\n", " ")
    try:
        note = notes["reviews"]
    except:
        note = [notes["document"]]

    if segment == "all":
        return [" ".join(note)]
    elif segment == "bestone":
        note = " ".join(note)
        return [s.text for s in nlp(note).sents]
    elif segment[:7] == "longest":
        num_sent = int(segment.split("_")[-1])
        note = " ".join(note)
        return segmentLongestSentence(note, num_sent)
    elif segment[:6] == "window":
        window_size = int(segment.split("_")[1])
        window_candidates = []
        for n in note:
            window_candidates.extend(segmentWindow(n, window_size))
        return window_candidates
    elif segment[:7] == "summary":
        note = " ".join(note)
        ratio = float(segment.split("_")[-1])
        return segmentSummary(note, ratio)
    elif segment[:10] == "extsummary":
        ratio = float(segment.split("_")[-1])
        return segmentExtSummary(notes['reviews'], notes['ext_summary'], ratio)
    elif segment[:4] == "bart":
        ratio = float(segment.split("_")[-1])
        return segmentBartSummary(notes['reviews'], notes['summary'], ratio)
    elif segment[:3] == "att":
        return [notes['att']]
    elif segment[:2] == "ig":
        return [notes['ig']]
    elif segment[:8] == "truncate":
        key = segment[9:]
        return [notes[key]]
    elif segment[:9] == "sentscore":
        key = segment[10:]
        # sentences = [s.text for s in nlp(notes[key]).sents]
        return notes[key]
    elif segment[:5] == "study":
        key = segment[6:]
        # sentences = [s.text for s in nlp(notes[key]).sents]
        return [" ".join(notes[key])]


def segmentLongestSentence(note, num_sent):
    sentences = [s.text for s in nlp(note).sents]
    lengths = [len(s) for s in sentences]
    # idx = np.argmax(lengths)
    idxs = np.argsort(lengths)[::-1][:num_sent]
    bestsent = " ".join([sentences[i] for i in idxs])
    return [bestsent]


def segmentWindow(note, w_size):
    sentences = [s.text for s in nlp(note).sents]
    if len(sentences) < w_size:
        return [" ".join(sentences)]
    return [" ".join(sentences[i:i+w_size]) for i in range(len(sentences)-w_size+1)]

def segmentSummary(note, ratio):
    summary = summarizer.summarize(note, ratio=ratio)
    return [summary]


def segmentExtSummary(reviews, summary, ratio):
    return [" ".join(summary)]


def segmentBartSummary(reviews, summary, ratio):
    return summary


def build_candidate_in_order(sentences, search_path):
    """
    try to reconstruct selected candidate sentences into original order
    """
    original_order = sorted(search_path)
    return " ".join([sentences[i] for i in original_order])


def get_cosine_similarity_matrix(source_sentences, target_sentences, model, business):
    """
    return N by N matrix of cosine similarity scores
    """
    import pathlib
    # generate cache filepath
    path = os.path.join(model.args.data_dir, "select_sentence", f"{model.args.data}", model.args.data_split, "similarity")
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    cached_file = "_".join(model.args.segment.split("_")[:5])  # ignore k alpha beta gamma
    cached_filepath = os.path.join(path, f"{cached_file}_{business}.npy")

    if not os.path.exists(cached_filepath):
        if "sentbert" in model.args.segment:
            sentbert = SentBert()
            source_embeddings = sentbert.encode(source_sentences, convert_to_numpy=True)
            target_embeddings = source_embeddings
        else:
            raise("sentence embedding is not in [tfidf, sentbert]")
        # [len of chunks * len of chunks]
        similarity = cosine_similarity(target_embeddings, source_embeddings)
        np.save(cached_filepath, similarity)
    else:
        similarity = np.load(cached_filepath, allow_pickle=True)
    
    return similarity


def get_self_cosine_similarity_matrix(source_sentences, cache_dir, business, method="sentbert"):
    """
    return N by N matrix of cosine similarity scores
    """
    cached_filepath = os.path.join(cache_dir, f"{business}.npy")

    if not os.path.exists(cached_filepath):
        if "sentbert" in method:
            sentbert = SentBert()
            source_embeddings = sentbert.encode(source_sentences, convert_to_numpy=True)
            target_embeddings = source_embeddings
        else:
            raise("sentence embedding is not in [tfidf, sentbert]")
        # [len of chunks * len of chunks]
        similarity = cosine_similarity(target_embeddings, source_embeddings)
        np.save(cached_filepath, similarity)
    else:
        similarity = np.load(cached_filepath, allow_pickle=True)
    
    return similarity


def get_sents_length_mask(chunks, businesses, cache_dir):
    cached_filepath = os.path.join(cache_dir, f"{businesses}.npy")
    if not os.path.exists(cached_filepath):
        # mask sentence lenght < 3
        sent_lengths = np.array([len(nlp(c.rstrip())) for c in chunks])
        sent_length_mask = np.where(sent_lengths < 3)[0]  # number of tokens
        np.save(cached_filepath, sent_length_mask)
    else:
        sent_length_mask = np.load(cached_filepath, allow_pickle=True)
    return sent_length_mask


def run_decsum(chunks, business, sent_preds, scores, preds_all_reviews,
                K, alpha, beta, gamma, model,
                query=None, beam_size=4):
    """
    Maximun marginal relevence for diverse chunk (window of sentences) selection
    @param 
        chunks : list of chunks (sentences from a same document)
        business: business id
        sent_preds: predicted scores of chunks based on a trained model (list of float)
        scores: scores for a given task. (First run of faithfulness score log(|x_s-y|))
        preds_all_reviews: predicted scores using all input
        K: output docs length
        alpha, beta, gamma: hyperparameters for three components, faithfulness, meaning diversity, and decision representiveness
        model: model class in `model_zoo.py`

    @return
        [concatenated selected sentences], [list of selected sentence orders]
        note that "concatenated selected sentences" is rearrange to original order
        , but "list of selected sentence orders" keeps the selected order
    """
    # for computing sentence similarity
    source_sentences = chunks
    target_sentences = chunks
    
    # mask sentence lenght < 3
    sent_lengths = np.array([len(nlp(c.strip())) for c in chunks])
    sent_length_mask = np.where(sent_lengths < 3)[0]  # number of tokens

    # compute cosine similarity for meaning diversity
    if beta > 0 :
        similarity = get_cosine_similarity_matrix(source_sentences, target_sentences, model, business)
    else:
        similarity = np.zeros((len(source_sentences), len(target_sentences)))

    # init beam search score
    if "WD" in model.args.segment:
        dist = []
        for pred in sent_preds:
            dist.append(wasserstein_distance([pred], sent_preds))
        dist = np.array(dist)

        # if decision representiveness is provided, use it
        if gamma != 0:
            updated_scores = gamma * -np.log(dist)
        # otherwise, use faithfulness socre
        else:
            updated_scores = scores
    else:
        # ad-hoc method
        NotImplementedError("No implemeted")

    candidates = target_sentences
    # skip sentence with length < 3
    candidate_paths = [[i] for i in range(len(candidates)) if i not in sent_length_mask]
    updated_scores = [score for i, score in enumerate(updated_scores) if i not in sent_length_mask]
        
    # select up to K chunks
    for iters in range(min(K-1, len(target_sentences)-1)):  # make sure length of target_sentences not smaller than k
        ranking = np.argsort(updated_scores)[::-1][:beam_size] # descending order
        
        # break if token length of current best candidate is longer than 50
        current_best_path = candidate_paths[ranking[0]]
        if model.args.trunc_length and sum(sent_lengths[current_best_path]) > model.args.trunc_length:
            # logger.info(f"Current best length {sum(sent_lengths[current_best_path])}")
            break

        paths = [candidate_paths[r] for r in ranking]
        beam_candidates = [candidates[r] for r in ranking]
        candidates = []
        candidate_paths = []
        max_similar = []  # sentence (semantically) repetitive penalty
        score_diversity = []  # value that represent the diversity of score
        
        for b in range(min(beam_size, len(beam_candidates))):
            p = paths[b]
            for i, s in enumerate(target_sentences):
                # make sure selected BEAM_SIZE sentences are not repeated
                if i not in paths[b] and i not in sent_length_mask:
                    candidates.append(build_candidate_in_order(target_sentences, p+[i]))
                    # candidates.append(
                    #     " ".join([target_sentences[idx] for idx in p+[i]])
                    # )
                    max_similar.append(similarity[i, paths[b]].max())
                    if "WD" in model.args.segment:  # use Wasserstein distance
                        add_one_path = paths[b] + [i]
                        selected_preds = sent_preds[add_one_path] 
                        dist = get_WD(selected_preds, sent_preds)
                        rating_score = -np.log(dist)
                        score_diversity.append(rating_score)
                    else:  # ad-hoc
                        NotImplementedError("No implemeted")
                    candidate_paths.append(p + [i])
         
        if model.args.data != "yelp":
            queries = [query] * len(candidates)
            candidates_run = (candidates, queries)
        else:
            # for multirc (deprecated)
            candidates_run = candidates

        # decision faithfulness
        if alpha != 0:
            preds = model.run_model(candidates_run)
            preds = np.array(preds)
            scores = -np.log(np.abs(preds-preds_all_reviews) + 1e-6)
        else:
            scores = np.zeros(preds_all_reviews.shape)

        # meaning diversity
        max_similar = np.array(max_similar)
        
        # decision representiveness
        score_diversity = np.array(score_diversity)

        updated_scores = alpha * scores - beta * max_similar + gamma * score_diversity

    best_idx = np.argmax(updated_scores)
    best_path = candidate_paths[best_idx]
    res = build_candidate_in_order(target_sentences, best_path)

    return [res], [best_path]


def get_WD(selec_dist, origin_dist):
    return wasserstein_distance(selec_dist, origin_dist)


def get_WDs(selected_idx, origin_dist):
    selected = origin_dist[selected_idx]
    dist = []
    for i in range(len(origin_dist)):
        if i in selected_idx:
            dist.append(1e6)
        else:
            tmp = list(selected) + [origin_dist[i]]
            tmp = np.array(tmp)
            dist.append(get_WD(tmp, origin_dist))
    return np.array(dist)


def score_func(preds, labels, mode='best'):
    if mode == 'best':
        return -np.abs(preds-labels)
    elif mode == 'highest':
        return preds
    elif mode == 'lowest':
        return -preds
    elif mode == 'order':
        # keep original order
        return -np.arange(len(preds))
    else:
        raise(f"score_func:{mode} does not exist")


def find_highest_score(preds, identities, segments, y_label, args,
                       top_k=1, true_label=None, sent_idx=None):
    dfs = []
    df = pd.DataFrame({'id':identities, 'pred':preds, 'segment':segments,
                        'y_label':y_label, 'sent_idx': sent_idx if sent_idx else [""]*len(identities)})
    unique_ids = df['id'].unique()
    for unique_id in unique_ids:
        tmp_df = df[df['id']==unique_id]
        tmp_df['score'] = score_func(tmp_df['pred'], tmp_df['y_label'], args.score_function)
        top_k_df = tmp_df.sort_values(by=['score'], ascending=False).iloc[:top_k] # from high score to low score
        dfs.append(top_k_df)
    
    result_df = pd.concat(dfs)

    best_scores = []
    best_sentences = []
    label = []
    ids = []
    for i, row in result_df.iterrows():
        ids.append(row['id'])
        best_scores.append(row['pred'])
        label.append(row['y_label'])
        best_sentences.append(row['segment'])
    if true_label:
        label = true_label
    
    if args.target_type == "reg":
        print("MSE")
        print(mean_squared_error(label, best_scores))
        score_type = "MSE"
        score = mean_squared_error(label, best_scores)
    else:
        print("F1")
        best_scores = [round(s) for s in best_scores]
        print(f1_score(label, best_scores))
        score_type = "F1"
        score = f1_score(label, best_scores)

    opt = args
    model_name = '.chkpt'
    if opt.feature_used == "all":
        model_name = "feature_text_" + model_name
    elif opt.feature_used == "all_but_notes":
        model_name = "feature_" + model_name
    else:
        model_name = "text_" + model_name

    import pathlib
    path = f'{args.result_dir}/results/{args.data}/{args.num_review}reviews/{args.data_split}/{args.model}/{args.segment}/{args.score_function}/{top_k}'
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(path, model_name[:-6]+".csv"), 'w') as f:
        f.write(f"TYPE,{score_type}\n")
        f.write(f"test,{score}")

    df = pd.DataFrame({"business":ids, "y_label":label, "pred": best_scores, "bestSents":best_sentences, "sent_idx":sent_idx})
    print(df)
    path = f'{args.result_dir}/selected_sentence/{args.data}/{args.num_review}reviews/{args.data_split}/{args.model}/{args.segment}/{args.score_function}/{top_k}'
    import pathlib
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    suffix = ""

    file_path = os.path.join(path, model_name[:-6] + f"{suffix}.csv")
    print(file_path)
    df.to_csv(file_path)

