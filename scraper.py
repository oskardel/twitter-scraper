import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import json
from requests import get
import random
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore 
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Embedding, Dense, LSTM # type: ignore
from tensorflow.keras.activations import relu, softmax # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.losses import CategoricalCrossentropy # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

def generate_ai_tweets(twitter_username: str, tweets_to_scrape:int = 200, tweets_to_generate: int = 1, epochs:int = 20):
    tweets = []
    first_scroll = True

    # Change these in the headers.json file by searching the HTTP request in Twitter and copying the headers mentioned below
    with open('./headers.json') as file:
        headers_credentials = json.load(file)

    headers = {
        "authorization" : headers_credentials["authorization"],
        "Cookie" : headers_credentials["Cookie"],
        "x-csrf-token" : headers_credentials["x-csrf-token"]
    }

    parameters = {
        "tweet_mode" : "extended",
        "tweet_search_mode": "live",
        "send_error_codes" : "true"
    }

    user_info = get(url='https://twitter.com/i/api/graphql/qW5u-DAuXpMEG0zA1F7UGQ/UserByScreenName?variables={"screen_name":"'+twitter_username+'","withSafetyModeUserFields":true}&features={"hidden_profile_likes_enabled":true,"hidden_profile_subscriptions_enabled":true,"rweb_tipjar_consumption_enabled":true,"responsive_web_graphql_exclude_directive_enabled":true,"verified_phone_label_enabled":false,"subscriptions_verification_info_is_identity_verified_enabled":true,"subscriptions_verification_info_verified_since_enabled":true,"highlights_tweets_tab_ui_enabled":true,"responsive_web_twitter_article_notes_tab_enabled":true,"creator_subscriptions_tweet_preview_api_enabled":true,"responsive_web_graphql_skip_user_profile_image_extensions_enabled":false,"responsive_web_graphql_timeline_navigation_enabled":true}&fieldToggles={"withAuxiliaryUserLabels":false}', headers=headers, params=parameters).json()
    user_id = user_info["data"]["user"]["result"]["rest_id"]

    # Scraping tweets from an account
    while len(tweets) < tweets_to_scrape:
        if first_scroll:
            twitter_request = get(url='https://twitter.com/i/api/graphql/ImwMsY2nbn1qxRNcUNL2DA/UserTweetsAndReplies?variables={"userId":"'+user_id+'","count":20,"includePromotedContent":true,"withCommunity":true,"withVoice":true,"withV2Timeline":true}&features={"rweb_tipjar_consumption_enabled":true,"responsive_web_graphql_exclude_directive_enabled":true,"verified_phone_label_enabled":false,"creator_subscriptions_tweet_preview_api_enabled":true,"responsive_web_graphql_timeline_navigation_enabled":true,"responsive_web_graphql_skip_user_profile_image_extensions_enabled":false,"communities_web_enable_tweet_community_results_fetch":true,"c9s_tweet_anatomy_moderator_badge_enabled":true,"articles_preview_enabled":true,"tweetypie_unmention_optimization_enabled":true,"responsive_web_edit_tweet_api_enabled":true,"graphql_is_translatable_rweb_tweet_is_translatable_enabled":true,"view_counts_everywhere_api_enabled":true,"longform_notetweets_consumption_enabled":true,"responsive_web_twitter_article_tweet_consumption_enabled":true,"tweet_awards_web_tipping_enabled":false,"creator_subscriptions_quote_tweet_preview_enabled":false,"freedom_of_speech_not_reach_fetch_enabled":true,"standardized_nudges_misinfo":true,"tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled":true,"tweet_with_visibility_results_prefer_gql_media_interstitial_enabled":true,"rweb_video_timestamps_enabled":true,"longform_notetweets_rich_text_read_enabled":true,"longform_notetweets_inline_media_enabled":true,"responsive_web_enhance_cards_enabled":false}&fieldToggles={"withArticlePlainText":false}', headers=headers, params=parameters)
            first_scroll = False
        else:
            twitter_request = get(url='https://twitter.com/i/api/graphql/ImwMsY2nbn1qxRNcUNL2DA/UserTweetsAndReplies?variables={"userId":"'+user_id+'","count":20,"cursor":"'+parameters["cursor"]+'","includePromotedContent":true,"withCommunity":true,"withVoice":true,"withV2Timeline":true}&features={"rweb_tipjar_consumption_enabled":true,"responsive_web_graphql_exclude_directive_enabled":true,"verified_phone_label_enabled":false,"creator_subscriptions_tweet_preview_api_enabled":true,"responsive_web_graphql_timeline_navigation_enabled":true,"responsive_web_graphql_skip_user_profile_image_extensions_enabled":false,"communities_web_enable_tweet_community_results_fetch":true,"c9s_tweet_anatomy_moderator_badge_enabled":true,"articles_preview_enabled":true,"tweetypie_unmention_optimization_enabled":true,"responsive_web_edit_tweet_api_enabled":true,"graphql_is_translatable_rweb_tweet_is_translatable_enabled":true,"view_counts_everywhere_api_enabled":true,"longform_notetweets_consumption_enabled":true,"responsive_web_twitter_article_tweet_consumption_enabled":true,"tweet_awards_web_tipping_enabled":false,"creator_subscriptions_quote_tweet_preview_enabled":false,"freedom_of_speech_not_reach_fetch_enabled":true,"standardized_nudges_misinfo":true,"tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled":true,"tweet_with_visibility_results_prefer_gql_media_interstitial_enabled":true,"rweb_video_timestamps_enabled":true,"longform_notetweets_rich_text_read_enabled":true,"longform_notetweets_inline_media_enabled":true,"responsive_web_enhance_cards_enabled":false}&fieldToggles={"withArticlePlainText":false}', headers=headers, params=parameters)
        
        if twitter_request.status_code != 200:
            twitter_request = get(url='https://twitter.com/i/api/graphql/9zyyd1hebl7oNWIPdA8HRw/UserTweets?variables={"userId":"1375205895345426433","count":20,"includePromotedContent":true,"withQuickPromoteEligibilityTweetFields":true,"withVoice":true,"withV2Timeline":true}&features={"rweb_tipjar_consumption_enabled":true,"responsive_web_graphql_exclude_directive_enabled":true,"verified_phone_label_enabled":false,"creator_subscriptions_tweet_preview_api_enabled":true,"responsive_web_graphql_timeline_navigation_enabled":true,"responsive_web_graphql_skip_user_profile_image_extensions_enabled":false,"communities_web_enable_tweet_community_results_fetch":true,"c9s_tweet_anatomy_moderator_badge_enabled":true,"articles_preview_enabled":true,"tweetypie_unmention_optimization_enabled":true,"responsive_web_edit_tweet_api_enabled":true,"graphql_is_translatable_rweb_tweet_is_translatable_enabled":true,"view_counts_everywhere_api_enabled":true,"longform_notetweets_consumption_enabled":true,"responsive_web_twitter_article_tweet_consumption_enabled":true,"tweet_awards_web_tipping_enabled":false,"creator_subscriptions_quote_tweet_preview_enabled":false,"freedom_of_speech_not_reach_fetch_enabled":true,"standardized_nudges_misinfo":true,"tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled":true,"tweet_with_visibility_results_prefer_gql_media_interstitial_enabled":true,"rweb_video_timestamps_enabled":true,"longform_notetweets_rich_text_read_enabled":true,"longform_notetweets_inline_media_enabled":true,"responsive_web_enhance_cards_enabled":false}&fieldToggles={"withArticlePlainText":false}', headers=headers, params=parameters)

        all_tweets_raw = twitter_request.json()

        for tweet in all_tweets_raw["data"]["user"]["result"]["timeline_v2"]["timeline"]["instructions"][-1]["entries"]:
            if "cursor" not in tweet["entryId"]:
                content = tweet.get("content", {})
                items = content.get("items")
                if items:
                    if "tweet_results" in items[-1]["item"]["itemContent"] and "legacy" in items[-1]["item"]["itemContent"]["tweet_results"]["result"]:
                        tweets.append(items[-1]["item"]["itemContent"]["tweet_results"]["result"]["legacy"]["full_text"])
                        print(items[-1]["item"]["itemContent"]["tweet_results"]["result"]["legacy"]["full_text"])
                    else:
                        continue
                else:
                    if "tweet_results" in content and "legacy" in content["tweet_results"]["result"]:
                        tweets.append(content["tweet_results"]["result"]["legacy"]["full_text"])
                        print(content["tweet_results"]["result"]["legacy"]["full_text"])
                    else:
                        continue

        for object in all_tweets_raw["data"]["user"]["result"]["timeline_v2"]["timeline"]["instructions"][-1]["entries"]:
            if "cursor-bottom" in object["entryId"]:
                parameters["cursor"] = object["content"]["value"]

    if len(tweets) > tweets_to_scrape:
        tweets = tweets[:tweets_to_scrape]

    # Creating the NLP Tokenizer
    tokenizer = Tokenizer(filters='()"', lower=True, oov_token="<OOV>")
    tokenizer.fit_on_texts(tweets)
    word_index = tokenizer.word_index
    vocab_size = len(word_index) + 1
    tokenized_tweets = tokenizer.texts_to_sequences(tweets)

    split_tweets = []
    for tweet in tokenized_tweets:
        for i in range(1, len(tokenized_tweets)):
            split_tweets.append(tweet[:i + 1])
        
    padded_tweets = pad_sequences(split_tweets, padding="pre", truncating="pre")

    tweet_inputs = padded_tweets[:, :-1]
    tweet_labels = to_categorical(padded_tweets[:, -1], num_classes=vocab_size)

    # Creating the machine learning model and training it
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=100, mask_zero=True),
        LSTM(units=128),
        Dense(units=64, activation=relu),
        Dense(units=vocab_size, activation=softmax)
    ])

    early_stopping = EarlyStopping(monitor="accuracy", patience=2)

    model.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=["accuracy"])
    model.fit(x=tweet_inputs, y=tweet_labels, epochs=epochs, batch_size=64, callbacks=early_stopping)
    
    # Creating a prediction    
    generated_tweets_list = []    
    for _ in range(tweets_to_generate):
        random_first_tweet = random.choice(tweets).split(" ")
        tweet_list = [random_first_tweet[0], random_first_tweet[1]]
        tweet_word_list = [tweet_list]
        
        while len(tweet_word_list[0]) < random.randint(5, 20):
            input = pad_sequences(tokenizer.texts_to_sequences(tweet_word_list), maxlen=vocab_size, padding="pre")
            word_prediction = model.predict(input)
            new_tokenized_word = np.argmax(word_prediction, axis=-1)[0]
            new_word = list(word_index.keys())[list(word_index.values()).index(new_tokenized_word)]
            tweet_word_list[0].append(new_word)
        
        new_tweet = " ".join(tweet_word_list[0])
        generated_tweets_list.append(new_tweet)

    return generated_tweets_list


if __name__ == "__main__":
    generated_tweets = generate_ai_tweets(twitter_username="eldenring", tweets_to_scrape=250, tweets_to_generate=5, epochs=25)
    print(f"List of tweets generated: {generated_tweets}")