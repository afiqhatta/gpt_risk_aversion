import openai
import pandas as pd
import tqdm
from openai.embeddings_utils import get_embedding, cosine_similarity
from ratelimit import limits, RateLimitException
import logging
from backoff import on_exception, expo
from params.rate_limits import OPENAI_RATE_LIMIT
import os
from keys import OPENAI_KEY

pd.options.mode.chained_assignment = None
openai.api_key = OPENAI_KEY


@on_exception(expo, RateLimitException, max_tries=8)
@limits(calls=OPENAI_RATE_LIMIT['calls'], period=OPENAI_RATE_LIMIT['period'])
def _get_embedding(
        text : str,
        max_token_length : int = 1000,
        model="text-davinci-002""
):
    """
    A function that calls an completion model on some text.
    :param text: Text to embed.
    :param model: Choice of model
    :return: Vector of similarites
    """
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']
