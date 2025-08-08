import vertexai
import os, sys
import json
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from methods import read_json_lines, write_json_lines, print_similar_videos



from vertexai.vision_models import MultiModalEmbeddingModel, Video
from vertexai.vision_models import VideoSegmentConfig

# TODO(developer): Update & uncomment line below
PROJECT_ID = "my-vertexai-project-id"
# Set the environment variable within the script
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/soujanyalanka/github/video-RAG/my-vertexai-project-id-4f210b8cc8e7.json"

vertexai.init(project=PROJECT_ID, location="us-central1")
video_list_filename = "video-list-for-index.txt"
bucket_url = "" #"gs://barc_videos/chunk_videos/"
model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")

index_jsonl_file = "video_embeddings.jsonl"

if os.path.exists(index_jsonl_file):
    print(f"The file '{index_jsonl_file}' exists.")
    vmetadata = read_json_lines(index_jsonl_file)
    # Create the DataFrame
    emb_df = pd.DataFrame(vmetadata)
    print("Created dataframe")

    query_index = 7
    print(vmetadata[query_index]["id"])
    print_similar_videos(vmetadata[query_index]['embedding'], emb_df)
    sys.exit(0)
else:
    print(f"The file '{index_jsonl_file}' does not exist.")


with open(video_list_filename) as vfp:
    video_list = vfp.read().splitlines()
    
    vembeddings = []
    vmetadata = []
    for video in video_list:
        embeddings = model.get_embeddings(
            video=Video.load_from_file(
            bucket_url + video,
            ),
            video_segment_config=VideoSegmentConfig(),
        )
        #print(embeddings)
        
        for j, vemb in enumerate(embeddings.video_embeddings):
            metadata = {}
            print("Embedding ", j)
            metadata['emb_position'] = j
            metadata['video_name'] = video
            metadata['id'] = video.split('/')[-1] + "_" + str(j)
            metadata['start_offset_sec'] = vemb.start_offset_sec
            metadata['end_offset_sec'] = vemb.end_offset_sec
            metadata['embedding_dims'] = len(vemb.embedding)
            metadata['embedding'] = vemb.embedding
            vembeddings.append(vemb.embedding)
            vmetadata.append(metadata)
            print("Start: ", vemb.start_offset_sec, "End: ", vemb.end_offset_sec)
            print(vemb.embedding[:10])
            print('Size of dims: ', len(vemb.embedding))

    # Create the DataFrame
    emb_df = pd.DataFrame(vmetadata)
    print("Created dataframe")
    write_json_lines(vmetadata, index_jsonl_file)
    print("Created jsonl file")

    print_similar_videos(vembeddings[4], emb_df)
    print()
    print(vmetadata[4]["id"])
    print_similar_videos(vmetadata[4]['embedding'], emb_df)