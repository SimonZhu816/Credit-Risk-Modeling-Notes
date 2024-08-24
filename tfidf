import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 示例数据
data = {
    'user_id': [1, 2, 3, 4, 5],
    'applist': [
        'Facebook Instagram WhatsApp',
        'Instagram Snapchat TikTok',
        'Facebook TikTok Twitter',
        'Facebook Instagram Twitter',
        'Snapchat TikTok Twitter'
    ],
    'credit_risk': ['low', 'high', 'low', 'high', 'low']  # 'low' 表示低风险, 'high' 表示高风险
}

df = pd.DataFrame(data)

# 初始化TF-IDF向量化器
vectorizer = TfidfVectorizer()

# 计算TF-IDF矩阵
tfidf_matrix = vectorizer.fit_transform(df['applist'])

# 获取词汇表
feature_names = vectorizer.get_feature_names()

# 将TF-IDF矩阵转换为DataFrame以便查看
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

# 标签编码
label_encoder = LabelEncoder()
df['credit_risk'] = label_encoder.fit_transform(df['credit_risk'])

# 分离特征和标签
X = df_tfidf
y = df['credit_risk']

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建分类模型
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练模型
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=label_encoder.classes_))
