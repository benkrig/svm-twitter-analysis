from wordcloud import WordCloud, STOPWORDS , ImageColorGenerator
import matplotlib.pyplot as plt


def get_worldcloud(tweet_df):
    # Start with one review:
    df_ADR = tweet_df[tweet_df.target == "1"]
    df_NADR = tweet_df[tweet_df.target == "0"]
    tweet_All = " ".join(review for review in tweet_df.text)
    tweet_ADR = " ".join(review for review in df_ADR.text)
    tweet_NADR = " ".join(review for review in df_NADR.text)

    # Create and generate a word cloud image:
    wordcloud_ALL = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(tweet_All)

    # Display the generated image:
    plt.imshow(wordcloud_ALL, interpolation='bilinear')
    plt.axis("off")
    plt.margins(x=0, y=0)
    plt.show()
