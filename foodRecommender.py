import pandas as pd
patients = pd.read_csv('patientsCSV.csv', sep=',', encoding='latin-1',
                    usecols=['Id', 'Diet'])
food = pd.read_csv('FoodTest.csv', sep=',', encoding='latin-1',
                     usecols=['FoodId', 'Name', 'Diet'])

# Check the top 5 rows
print(patients.head())
# Check the file info
print(patients.info())
# Check the top 5 rows
print(food.head())
# Check the file info
print(food.info())


# Make a census of the diet keywords
diet_labels = set()
for s in food['Diet'].str.split(',').values:
    diet_labels = diet_labels.union(set(s))


#Content based
# Break up the big diet string into a string array
food['Diet'] = food['Diet'].str.split(',')
# Convert diet to string value
food['Diet'] = food['Diet'].fillna("").astype('str')


from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer='char',ngram_range=(1, 2),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(food['Diet'])
tfidf_matrix.shape
from sklearn.metrics.pairwise import linear_kernel
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim[:4, :4]
# Build a 1-dimensional array with food names
titles = food['Name']
indices = pd.Series(food.index, index=food['Diet'])
# Function that get food recommendations based on the cosine similarity score of food diet
def food_recommendations(diet):
    idx = indices[diet]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    food_indices = [i[0] for i in sim_scores]
    return titles.iloc[food_indices]
# Top 5 similar food:
print (food_recommendations(1).head(5))
