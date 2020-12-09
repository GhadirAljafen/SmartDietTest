import pandas as pd
food = pd.read_csv('FoodDataset.csv', sep=',', encoding='latin-1',
                   usecols=['ID', 'Food', 'MealType', 'Serving', 'Calories', 'Diet', 'Units'])
# Check the top 5 rows
print(food.head())
# Check the file info
print(food.info())
diet_labels = set()
for s in food['Diet'].str.split('|').values:
    diet_labels = diet_labels.union(set(s))


# Content based
food['Diet'] = food['Diet'].str.split('|')
food['Diet'] = food['Diet'].fillna("").astype('str')

from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer='char', ngram_range=(1, 2), min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(food['Diet'])
tfidf_matrix.shape

from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim[:4, :4]
# Build a 1-dimensional array with food names
titles = food['ID']
indices = pd.Series(food.index, index=food['Food'])
print(indices)
# Function that get food recommendations based on the cosine similarity score of food diet
def food_recommendations(diet):
    idx = indices[diet]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    food_indices = [i[0] for i in sim_scores]
    return titles.iloc[food_indices]
# Top 20 similar food:
print('Recommended Food:')
print(food_recommendations(3).head(20))



Diets = food_recommendations(0).head(20)

import pyodbc

def Insert(conn):
     print("Insert into Recommendation:")
     cursor = conn.cursor()
     for x in Diets:
        cursor.execute('insert into Recommendation(FoodId,DietType) values(?,?);',(x,0))
     conn.commit()

conn = pyodbc.connect(

    "Driver={SQL Server Native Client 11.0};"
    "Server=DESKTOP-IKUFVQO;"
    "Database=SmartDiet;"
    "Trusted_Connection=yes;"
)

Insert(conn)
conn.close()





def read(conn):
    print("Read")
    cursor = conn.cursor()
    cursor.execute("Select * from Food")
    for row in cursor:
        print(f'row={row}')
    print()
