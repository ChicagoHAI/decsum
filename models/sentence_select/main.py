
#!/usr/bin/env python3
import logging
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
import argparse
import models.sentence_select.model_zoo as model_zoo

logger = logging.getLogger(__name__)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_used', type=str, help='feature used',
                        default="notes", choices=["all", "notes", "all_but_notes"])
    parser.add_argument("--debug", action="store_true", default=False,
                        help="whether to enter debug mode")
    parser.add_argument('--segment', type=str, help='segmentation type')
    parser.add_argument('--data', type=str, help='dataset', default="yelp")
    parser.add_argument('--data_split', type=str, help='data split naming', default="test")
    parser.add_argument('--target_type', type=str, help='target type. regression or classification', default="reg", choices=["reg", "cls"])
    parser.add_argument('--score_function', type=str, help='score function for choosing return segments', default="best")
    parser.add_argument('--return_candidates', type=int, help='', default=1)
    parser.add_argument('--model', type=str, help='task',
                        choices=["Transformer"])
    parser.add_argument('--device', type=str, help='task',
                        default="0")
    parser.add_argument('--num_review', type=int, help='task',
                        default=50)
    parser.add_argument('--num_sentences', type=str, 
                        help='number of sentences selected in the summary',
                        default="6")
    parser.add_argument('--alpha', type=float,
                        help="hyperparameter for decision faithfulness",
                        default=1)
    parser.add_argument('--beta', type=float,
                        help="hyperparameter for semantic diversity",
                        default=1)
    parser.add_argument('--gamma', type=float,
                        help="hyperparameter for decision representativeness",
                        default=1)
    parser.add_argument('--trunc_length', type=int,
                        help="hyperparameter for language fluency",
                        default=None)
    parser.add_argument('--data_dir', type=str, help='task',
                        default="/data/joe/Information-Solicitation-Yelp/")
    parser.add_argument('--result_dir', type=str, help='result directory',
                        default="/data/joe/Information-Solicitation-Yelp/models/sentence_select/")
    parser.add_argument('--trained_model_path', type=str, help='task', required=True)
    args = parser.parse_args()

    # convert truncation to sentence length
    if "trunc" in args.num_sentences:
        args.trunc_length = int(args.num_sentences[:-5])
        args.num_sentences = int(args.num_sentences[:-5])
    else:
        args.num_sentences = int(args.num_sentences)
    logger.info(args)

    if args.model == "Transformer":
        model = model_zoo.Transformer(args)
    else:
        raise(f"Model {args.model} does not exist.")
    model.predict()
