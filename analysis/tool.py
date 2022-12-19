import nltk
import numpy as np
from scipy import stats
from prettytable import PrettyTable
import statsmodels.api as sm

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

def get_scores(df, source):
    t = PrettyTable(['Category', 'Avg Score'])
    # the score for replies
    t.add_row(['Tone', df[df['source']==source]["Tone"].mean()])
    t.add_row(['Hate Speech', df[df['source']==source]["HS"].mean()])
    t.add_row(['Offensive Language', df[df['source']==source]["OL"].mean()])
    t.add_row(['Negative Emotion', df[df['source']==source]["emo_neg"].mean()])
    t.add_row(['Anxiety', df[df['source']==source]["emo_anx"].mean()])
    t.add_row(['Anger', df[df['source']==source]["emo_anger"].mean()])
    t.add_row(['Sadness', df[df['source']==source]["emo_sad"].mean()])
    t.add_row(['Swear', df[df['source']==source]["swear"].mean()])
    print(t)
    # check the percetage of negtive tone of the replies
    count = df[df['source']==source]["Tone"][df[df['source']==source]["Tone"] < 50].count()
    print("the percetage of negtive tone:", count/len(df[df['source']==source]))
    

def ttest(df1, df2):
    # t-test
    t = PrettyTable(['Category', 'T-value', 'P-value'])

    tone = stats.ttest_ind(df1.Tone, df2.Tone)
    t.add_row(['Tone', tone[0], tone[1]])

    hs = stats.ttest_ind(df1.HS, df2.HS)
    t.add_row(['Hate Speech', hs[0], hs[1]])

    ol = stats.ttest_ind(df1.OL, df2.OL)
    t.add_row(['Offensive Language', ol[0], ol[1]])

    neg = stats.ttest_ind(df1.emo_neg, df2.emo_neg)
    t.add_row(['Negative Emotion', neg[0], neg[1]])

    anx = stats.ttest_ind(df1.emo_anx, df2.emo_anx)
    t.add_row(['Anxiety', anx[0], anx[1]])

    anger = stats.ttest_ind(df1.emo_anger, df2.emo_anger)
    t.add_row(['Anger', anger[0], anger[1]])

    sad = stats.ttest_ind(df1.emo_sad, df2.emo_sad)
    t.add_row(['Sadness', sad[0], sad[1]])

    swear = stats.ttest_ind(df1.swear, df2.swear)
    t.add_row(['Swear', swear[0], swear[1]])
    print(t)

def OSL_result_overall(df):
    df_endog = df.index.to_julian_date()
    df_exog = df[['Tone', 'HS', 'OL', 'emo_neg', 'emo_anx', 'emo_anger', 'emo_sad', 'swear']]

    df_exog = sm.add_constant(df_exog)
    model = sm.OLS(df_endog, df_exog)
    results = model.fit()

    return results

def OSL_result(df, category):
    df_endog = df.index.to_julian_date()
    df_exog = df[category]

    df_exog = sm.add_constant(df_exog)
    model = sm.OLS(df_endog, df_exog)
    results = model.fit()

    return results

words = set(nltk.corpus.words.words())
def remove_misspelling_words(text):
    text = str(text)
    return " ".join(w for w in nltk.wordpunct_tokenize(text) \
         if w.lower() in words or not w.isalpha())

# def linear_model_status(df, category):
#     print("The statistical significance of the replies is:")
#     df = df.sort_values(by=['date'])
#     y = [i for i in df.groupby('date')[category].mean()]
#     x = np.arange(len(y))

#     date = np.array(df['date'].unique().tolist(), dtype='datetime64')
#     date.sort()

#     res = stats.linregress(x, y)
#     res.predict(x)

#     print(f"R-squared: {res.rvalue**2:.6f}")
#     print(f"p-value: {res.pvalue:.6f}")
#     print(f"standard error: {res.stderr:.6f}")
#     print(f"intercept: {res.intercept:.6f}")
#     print(f"slope: {res.slope:.6f}")

#     # plot the linear model
#     f, ax = plt.subplots()
#     f.set_size_inches(18, 4)
#     ax.xaxis.update_units(date)
#     # ax.set_xticklabels(date, rotation=90)
#     ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m/%d'))
#     ax.xaxis.set_major_locator(mdates.DayLocator(interval=20))

#     plt.xticks(rotation=90)
#     plt.plot(date, y, 'o', label='daily average')
#     plt.plot(date, res.intercept + res.slope*x, 'r', label='fitted line')
#     plt.legend(loc=1)
#     f.tight_layout()
#     plt.show()

#     return res

