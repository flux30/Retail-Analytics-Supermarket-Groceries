import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from mlxtend.frequent_patterns import apriori, association_rules
from imblearn.over_sampling import SMOTE
import chrono
import os
from datetime import datetime
import matplotlib.patches as patches

sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 12

project_dir = "C:/Users/bijay/Documents/dmw_project1"
output_dir = os.path.join(project_dir, "outputs")
os.makedirs(output_dir, exist_ok=True)

def parse_date(date_str):
    if pd.isna(date_str) or not isinstance(date_str, str):
        return pd.NaT
    try:
        return chrono.parse_date(date_str)
    except:
        return pd.to_datetime(date_str, errors='coerce')

kpis = [
    "Total Sales Revenue",
    "Profit Margin",
    "Average Order Value",
    "Sales by Category",
    "Sales by Sub-Category",
    "Sales by Region",
    "Sales by City",
    "Customer Purchase Frequency",
    "Discount Impact on Sales",
    "Profit by Category"
]

file_path = "C:/Users/bijay/Documents/dmw_project1/DIY_1_RetailSupermarket/Supermart Grocery Sales - Retail Analytics Dataset.csv"
try:
    df = pd.read_csv(file_path)
    print(f"Dataset loaded successfully. Shape: {df.shape}")
    print("Columns:", df.columns.tolist())
except FileNotFoundError:
    print(f"Error: Dataset file not found at {file_path}")
    exit(1)
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

print(f"Initial rows: {len(df)}")
print("Missing values before preprocessing:\n", df.isnull().sum())
df['Sales'] = df['Sales'].fillna(df['Sales'].median())
df['Discount'] = df['Discount'].fillna(df['Discount'].median())
df['Profit'] = df['Profit'].fillna(df['Profit'].median())
df = df.dropna(
    subset=['Order ID', 'Customer Name', 'Category', 'Sub Category', 'City', 'Order Date', 'Region', 'State'])
print(f"Rows after handling missing values: {len(df)}")

df['Order Date'] = df['Order Date'].apply(parse_date)
df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
df['Discount'] = pd.to_numeric(df['Discount'], errors='coerce')
df['Profit'] = pd.to_numeric(df['Profit'], errors='coerce')

print("Missing values after type conversion:\n", df.isnull().sum())
df = df.dropna(subset=['Sales', 'Discount', 'Profit', 'Order Date'])
print(f"Rows after dropping NaN in Sales, Discount, Profit, Order Date: {len(df)}")


def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 3.0 * IQR
    upper_bound = Q3 + 3.0 * IQR
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
    print(f"Outliers in {column}: {len(outliers)}")
    return outliers

if len(df) > 0:
    sales_outliers = detect_outliers(df, 'Sales')
    profit_outliers = detect_outliers(df, 'Profit')
    df = df[~df['Sales'].isin(sales_outliers)]
    df = df[~df['Profit'].isin(profit_outliers)]
    print(f"Rows after removing outliers: {len(df)}")
else:
    print("Error: DataFrame is empty after preprocessing. Cannot proceed.")
    exit(1)

if len(df) == 0:
    print("Error: No data remains after preprocessing. Check dataset for issues.")
    exit(1)

df['Order Year'] = df['Order Date'].dt.year
df['Order Month'] = df['Order Date'].dt.month

total_sales = df['Sales'].sum()
print(f"Total Sales Revenue: {total_sales:.2f}")

df['Profit Margin'] = (df['Profit'] / df['Sales']) * 100
avg_profit_margin = df['Profit Margin'].mean()
print(f"Average Profit Margin: {avg_profit_margin:.2f}%")

avg_order_value = df['Sales'].mean()
print(f"Average Order Value: {avg_order_value:.2f}")

sales_by_category = df.groupby('Category')['Sales'].sum().sort_values(ascending=False)
if not sales_by_category.empty:
    plt.figure(figsize=(10, 6))
    sns.barplot(x=sales_by_category.index, y=sales_by_category.values, hue=sales_by_category.index,
                palette=sns.color_palette("husl", n_colors=len(sales_by_category)), legend=False)
    plt.title('Sales by Category', fontsize=14, weight='bold', color='#2F4F4F')
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Total Sales', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sales_by_category.png'), dpi=300)
    plt.close()
else:
    print("Warning: No data for Sales by Category plot.")

sales_by_subcategory = df.groupby('Sub Category')['Sales'].sum().sort_values(ascending=False)
if not sales_by_subcategory.empty:
    plt.figure(figsize=(12, 6))
    sns.barplot(x=sales_by_subcategory.index, y=sales_by_subcategory.values, hue=sales_by_subcategory.index,
                palette=sns.color_palette("husl", n_colors=len(sales_by_subcategory)), legend=False)
    plt.title('Sales by Sub-Category', fontsize=14, weight='bold', color='#2F4F4F')
    plt.xlabel('Sub-Category', fontsize=12)
    plt.ylabel('Total Sales', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sales_by_subcategory.png'), dpi=300)
    plt.close()
else:
    print("Warning: No data for Sales by Sub-Category plot.")

sales_by_region = df.groupby('Region')['Sales'].sum().sort_values(ascending=False)
if not sales_by_region.empty:
    plt.figure(figsize=(8, 6))
    sns.barplot(x=sales_by_region.index, y=sales_by_region.values, hue=sales_by_region.index,
                palette=sns.color_palette("blend:#4C78A8,#F58518", n_colors=len(sales_by_region)), legend=False)
    plt.title('Sales by Region', fontsize=14, weight='bold', color='#2F4F4F')
    plt.xlabel('Region', fontsize=12)
    plt.ylabel('Total Sales', fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sales_by_region.png'), dpi=300)
    plt.close()
else:
    print("Warning: No data for Sales by Region plot.")

sales_by_city = df.groupby('City')['Sales'].sum().sort_values(ascending=False).head(10)
if not sales_by_city.empty:
    plt.figure(figsize=(10, 6))
    sns.barplot(x=sales_by_city.index, y=sales_by_city.values, hue=sales_by_city.index,
                palette=sns.color_palette("husl", n_colors=len(sales_by_city)), legend=False)
    plt.title('Top 10 Cities by Sales', fontsize=14, weight='bold', color='#2F4F4F')
    plt.xlabel('City', fontsize=12)
    plt.ylabel('Total Sales', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sales_by_city.png'), dpi=300)
    plt.close()
else:
    print("Warning: No data for Sales by City plot.")

customer_frequency = df['Customer Name'].value_counts().head(10)
if not customer_frequency.empty:
    plt.figure(figsize=(10, 6))
    sns.barplot(x=customer_frequency.index, y=customer_frequency.values, hue=customer_frequency.index,
                palette=sns.color_palette("blend:#4C78A8,#F58518", n_colors=len(customer_frequency)), legend=False)
    plt.title('Top 10 Customers by Purchase Frequency', fontsize=14, weight='bold', color='#2F4F4F')
    plt.xlabel('Customer Name', fontsize=12)
    plt.ylabel('Number of Purchases', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'customer_frequency.png'), dpi=300)
    plt.close()
else:
    print("Warning: No data for Customer Purchase Frequency plot.")

if len(df) > 0:
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Discount', y='Sales', data=df, hue='Profit', size='Profit', palette='viridis')
    plt.title('Discount Impact on Sales and Profit', fontsize=14, weight='bold', color='#2F4F4F')
    plt.xlabel('Discount', fontsize=12)
    plt.ylabel('Sales', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'discount_impact.png'), dpi=300)
    plt.close()
else:
    print("Warning: No data for Discount Impact plot.")

profit_by_category = df.groupby('Category')['Profit'].sum().sort_values(ascending=False)
if not profit_by_category.empty:
    plt.figure(figsize=(10, 6))
    sns.barplot(x=profit_by_category.index, y=profit_by_category.values, hue=profit_by_category.index,
                palette=sns.color_palette("husl", n_colors=len(profit_by_category)), legend=False)
    plt.title('Profit by Category', fontsize=14, weight='bold', color='#2F4F4F')
    plt.xlabel('Category', fontsize=12)
    plt.ylabel('Total Profit', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'profit_by_category.png'), dpi=300)
    plt.close()
else:
    print("Warning: No data for Profit by Category plot.")

plt.figure(figsize=(12, 8))
plt.gca().add_patch(
    plt.Rectangle((0, 0), 1, 1, transform=plt.gca().transAxes, facecolor=sns.color_palette("Blues", as_cmap=True)(0.2)))
steps = [
    ("1. Load Dataset\n(CSV)", 0.95, sns.color_palette("husl", 5)[0]),
    ("2. Preprocess Data\n(Handle Missing, Convert Types, Remove Outliers)", 0.75, sns.color_palette("husl", 5)[1]),
    ("3. Exploratory Data Analysis\n(KPIs, Visualizations)", 0.55, sns.color_palette("husl", 5)[2]),
    ("4. Association Rule Mining\n(Apriori Algorithm)", 0.35, sns.color_palette("husl", 5)[3]),
    ("5. Classification Model & Data Warehouse\n(Random Forest, Non-Volatile Dataset)", 0.15,
     sns.color_palette("husl", 5)[4])
]
for text, y_pos, color in steps:
    plt.gca().add_patch(
        patches.FancyBboxPatch((0.15, y_pos - 0.05), 0.7, 0.1, boxstyle="round,pad=0.02", edgecolor='black',
                               facecolor=color, alpha=0.8))
    plt.text(0.5, y_pos, text, ha='center', fontsize=12, weight='bold', color='white')
    if y_pos < 0.95:
        plt.text(0.5, y_pos + 0.1, "↓", ha='center', fontsize=14, weight='bold', color='black')
plt.text(0.5, 1.0, "Supermart Retail Analytics Workflow", ha='center', fontsize=16, weight='bold', color='#2F4F4F')
plt.axis('off')
plt.savefig(os.path.join(output_dir, 'workflow_diagram.png'), dpi=300, bbox_inches='tight')
plt.close()

if len(df) > 0:
    basket = df.groupby(['Order ID', 'Sub Category'])['Sales'].count().unstack().reset_index().fillna(0)
    basket.set_index('Order ID', inplace=True)
    basket = basket.map(lambda x: 1 if x > 0 else 0).astype(bool)
    frequent_itemsets = apriori(basket, min_support=0.001, use_colnames=True)
    print(f"Number of frequent itemsets: {len(frequent_itemsets)}")
    if len(frequent_itemsets) > 0:
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.5)
        rules = rules.sort_values('lift', ascending=False).head(10)
        rules.to_csv(os.path.join(output_dir, 'association_rules.csv'))
        print("Top 10 Association Rules:")
        print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
    else:
        print("No frequent itemsets found. Try lowering min_support further.")
else:
    print("Warning: No data for Association Rule Mining.")

if len(df) > 0:
    le_category = LabelEncoder()
    le_subcategory = LabelEncoder()
    le_region = LabelEncoder()
    le_city = LabelEncoder()
    df['Category_Encoded'] = le_category.fit_transform(df['Category'])
    df['Sub Category_Encoded'] = le_subcategory.fit_transform(df['Sub Category'])
    df['Region_Encoded'] = le_region.fit_transform(df['Region'])
    df['City_Encoded'] = le_city.fit_transform(df['City'])
    X = df[['Sales', 'Profit', 'Discount', 'Region_Encoded', 'City_Encoded', 'Order Year', 'Order Month']]
    y_category = df['Category_Encoded']
    y_subcategory = df['Sub Category_Encoded']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(random_state=42)
    X_train_cat, y_train_cat = smote.fit_resample(X_scaled, y_category)
    X_train_subcat, y_train_subcat = smote.fit_resample(X_scaled, y_subcategory)

    X_train_cat, X_test_cat, y_train_cat, y_test_cat = train_test_split(X_train_cat, y_train_cat, test_size=0.2,
                                                                        random_state=42)
    X_train_subcat, X_test_subcat, y_train_subcat, y_test_subcat = train_test_split(X_train_subcat, y_train_subcat,
                                                                                    test_size=0.2, random_state=42)

    param_grid = {'n_estimators': [50, 100], 'max_depth': [10, 20, None]}
    rf_category = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)
    rf_category.fit(X_train_cat, y_train_cat)
    y_pred_cat = rf_category.predict(X_test_cat)

    rf_subcategory = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1)
    rf_subcategory.fit(X_train_subcat, y_train_subcat)
    y_pred_subcat = rf_subcategory.predict(X_test_subcat)

    cat_report = classification_report(y_test_cat, y_pred_cat, target_names=le_category.classes_)
    subcat_report = classification_report(y_test_subcat, y_pred_subcat, target_names=le_subcategory.classes_)
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write("Category Classification Report:\n")
        f.write(cat_report)
        f.write("\nSub-Category Classification Report:\n")
        f.write(subcat_report)
    print("Category Classification Report:")
    print(cat_report)
    print("Sub-Category Classification Report:")
    print(subcat_report)
else:
    print("Warning: No data for Classification Model.")

if len(df) > 0:
    df['Version'] = 1
    df['Update_Timestamp'] = datetime.now()
    modified_dataset_path = os.path.join(output_dir, 'supermart_data_warehouse.csv')
    df.to_csv(modified_dataset_path, index=False)


    def update_dataset(original_df, new_data_path, version):
        new_df = pd.read_csv(new_data_path)
        new_df['Order Date'] = new_df['Order Date'].apply(parse_date)
        new_df['Sales'] = pd.to_numeric(new_df['Sales'], errors='coerce')
        new_df['Discount'] = pd.to_numeric(new_df['Discount'], errors='coerce')
        new_df['Profit'] = pd.to_numeric(new_df['Profit'], errors='coerce')
        new_df = new_df.dropna(subset=['Sales', 'Discount', 'Profit', 'Order Date'])
        new_df['Version'] = version
        new_df['Update_Timestamp'] = datetime.now()
        updated_df = pd.concat([original_df, new_df], ignore_index=True)
        updated_df.to_csv(modified_dataset_path, index=False)
        return updated_df


    print(f"Modified dataset saved at: {modified_dataset_path}")
else:
    print("Warning: No data for Data Warehouse modification.")

enhancements = """
Enhancements Suggested:
1. **Preprocessing**: Used median imputation and added date-based features (Order Year, Month).
2. **Outlier Detection**: Relaxed IQR threshold to 3.0; suggest Z-score for robustness.
3. **Model Development**: Added City_Encoded, date features, SMOTE for class imbalance, and GridSearchCV.
4. **EDA**: Visualizations use Arial font and dynamic color palettes.
5. **Data Warehouse**: Implemented versioning and timestamp.
6. **Association Rules**: Lowered min_support to 0.001 to capture more rules.
"""
with open(os.path.join(output_dir, 'enhancements.txt'), 'w') as f:
    f.write(enhancements)
print("Enhancements suggested saved to enhancements.txt")