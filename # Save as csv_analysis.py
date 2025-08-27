# Save as csv_analysis.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
df = pd.read_csv('data.csv')   # Make sure data.csv is present in same directory

# Calculate average of a selected column, e.g., 'OrderAmountGBP'
average_order = df['OrderAmountGBP'].mean()
print("Average order amount:", average_order)

# Bar Chart: count of ProductGroup
df['ProductGroup'].value_counts().plot.bar()
plt.title('Orders per Product Group')
plt.ylabel('Number of Orders')
plt.xlabel('Product Group')
plt.tight_layout()
plt.show()

# Scatter Plot: Age vs OrderAmountGBP
plt.scatter(df['Age'], df['OrderAmountGBP'])
plt.xlabel('Age')
plt.ylabel('Order Amount')
plt.title('Age v/s Order Amount')
plt.tight_layout()
plt.show()

# Heatmap: Correlation
sns.heatmap(df.corr(), annot=True)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# Insights Example
print(df.corr())
# Save as house_price_pred.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load CSV Data
df = pd.read_csv('house_prices.csv')  # Use a Kaggle dataset or the below format

# Select features
X = df[['Avg. Area Number of Rooms', 'Avg. Area House Age', 'Area Population']]
y = df['Price']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Model Training
reg = LinearRegression()
reg.fit(X_train, y_train)

# Predict
predictions = reg.predict(X_test)
print("Sample Predictions:", predictions[:5])

# Evaluate
print("Model Score (R^2):", reg.score(X_test, y_test))

# Visualization: Actual vs Predicted
plt.scatter(y_test, predictions)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("House Price Prediction")
plt.tight_layout()
plt.show()
# Save as app.py
from flask import Flask, request, render_template_string
from textblob import TextBlob

app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Sentiment Analysis</title>
</head>
<body>
    <h2>Enter Text for Sentiment Analysis</h2>
    <form method="POST">
        <input type="text" name="usertext" required>
        <input type="submit" value="Analyze">
    </form>
    {% if result %}
      <h3>Result:</h3>
      <p>Sentiment: {{ result }}</p>
      <p>Polarity: {{ polarity }}</p>
      <p>Subjectivity: {{ subjectivity }}</p>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=['GET', 'POST'])
def home():
    result = polarity = subjectivity = ""
    if request.method == 'POST':
        text = request.form['usertext']
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        if polarity > 0:
            result = "Positive"
        elif polarity < 0:
            result = "Negative"
        else:
            result = "Neutral"
    return render_template_string(HTML,
                                 result=result,
                                 polarity=polarity,
                                 subjectivity=subjectivity)

if __name__ == "__main__":
    app.run(debug=True)
