"""
Data can be downloaded from https://www.yelp.com/dataset/download
"""

import argparse
import gzip
import json
import logging
import os
import pprint
from datetime import datetime

import pandas as pd

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logging.info('Admin logged in')

pp = pprint.PrettyPrinter(indent=4)
myprint = pp.pprint
DATE_PATTERN = "%Y-%m-%d %H:%M:%S"


def get_created_time(text):
    return int(datetime.strptime(text, DATE_PATTERN).strftime("%S"))


def convert_data(input_file, output_file):
    with open(input_file) as fin, gzip.open(output_file, "wt", encoding='utf-8') as fout:
        business_dict = {}
        count = 0
        for line in fin:
            data = json.loads(line)
            business_id = data["business_id"]
            if business_id not in business_dict:
                business_dict[business_id] = []
            count += 1
            business_dict[business_id].append((
                get_created_time(data["date"]),
                data["user_id"],
                data["text"],
                data["stars"]))
            if count % 100000 == 0:
                logging.info(count)
        for b in business_dict:
            business_dict[b].sort()
            fout.write("%s\n" % json.dumps({"business": b, "reviews": business_dict[b]}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--yelp_data_dir",
                        default="data/yelp/",
                        type=str,
                        help="")
    parser.add_argument("--output_dir",
                        default="data/out/",
                        type=str,
                        help="")
    parser.add_argument("--data_split",
                        action='store_true', 
                        default=True,
                        help="")
    parser.add_argument("--num_review",
                        type=int,
                        default=50,
                        help="Number of reviews for computing average rating")    
    
    args = parser.parse_args() 
    logging.info(f"args: {args}")   

    restaurant_business_ids = set()
    with open(os.path.join(args.yelp_data_dir, "business.json"), "r", encoding='utf-8') as f:
        for line in f:  
            if line:    
                json_content = json.loads(line)
                if json_content["categories"] != None:                        
                    categories = [val.lower().strip() for val in json_content["categories"].split(",")]
                    if "restaurants" in categories:
                        restaurant_business_ids.add(json_content["business_id"])
    
    logging.info(f"Finished reading the business.json file. There are {len(restaurant_business_ids)} unique restaurants in the dataset.")

    grouped_reviews_filepath = os.path.join(args.yelp_data_dir, "grouped_reviews.jsonlist.gz")
    if not os.path.exists(grouped_reviews_filepath):
        logging.info(f'Grouped reviews not exist. Building grouped reviews at {grouped_reviews_filepath}')
        convert_data(os.path.join(args.yelp_data_dir, "review.json"), grouped_reviews_filepath)
        logging.info(f'Grouped reviews file built')

    logging.info(f"converting grouped reviews into resutaurant only reviews and compute average of first {args.num_review} reviews")
    out_file = os.path.join(args.output_dir, f"yelp_10reviews_{args.num_review}avg.jsonl.gz")
    with gzip.open(grouped_reviews_filepath, 'rt', encoding='utf-8') as f_in,  gzip.open(out_file, "wt", encoding='utf-8') as fout:
        for l in f_in:
            r = json.loads(l)
            if r['business'] not in restaurant_business_ids:
                continue
            if len(r['reviews']) >=args.num_review:
                tmp = {"reviews":[], "scores":[]}
                tmp['business'] = r['business']
                for i in range(10):
                    tmp["reviews"].append(r["reviews"][i][2])
                    tmp["scores"].append(r["reviews"][i][3])
                tmp["avg_score"] = sum([r["reviews"][j][3] for j in range(args.num_review)])/args.num_review
                fout.write("%s\n" % json.dumps(tmp))

    if args.data_split:

        split_datapath = os.path.join(args.output_dir, f"{args.num_review}reviews")
        if not os.path.exists(split_datapath):
            logging.info(f'Split datapath not exist. Create split data dir at {split_datapath}')
            os.makedirs(split_datapath, exist_ok=True)
        
        def dump_jsonl_gz(obj, outpath):
            # obj is list of json
            with gzip.open(outpath, "wt", encoding='utf-8') as fout:
                for o in obj:
                    fout.write("%s\n" % json.dumps(o))
        
        splits = ["train", "dev", "test"]
        split_ids = {}
        for s in splits:
            split_ids[s] = set(pd.read_csv(f"preprocess/{s}_business_ids.csv").business.values)
        
        reviews = []
        with gzip.open(out_file, 'rt', encoding='utf-8') as f:
            for line in f:
                reviews.append(json.loads(line))
            
        for split, ids in split_ids.items():
            split_reviews = []
            for review in reviews:
                if review['business'] in ids:
                    split_reviews.append(review)

            storepath = os.path.join(split_datapath, f"{split}.jsonl.gz")
            dump_jsonl_gz(split_reviews, storepath)
            logging.info(f"Data split length: {split} ({len(split_reviews)}), stored at: {storepath}")

    logging.info("Finished!")
