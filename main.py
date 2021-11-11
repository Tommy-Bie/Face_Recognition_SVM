import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # 数据集划分函数
from sklearn.datasets import fetch_lfw_people  # 人脸数据集
from sklearn.model_selection import GridSearchCV  # 网格搜索交叉验证
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA  # 主成分分析
from sklearn.svm import SVC  # 支持向量机分类器

# 数据读取
face = fetch_lfw_people(min_faces_per_person=70, resize=0.4)  # 选取样本数不少于70的类别
print('原始数据维度：', face.data.shape)
num_samples, h, w = face.images.shape
X = face.data  # 数据
y = face.target  # 人脸标签
target_names = face.target_names  # 人名（类别名）
num_features = X.shape[1]  # 特征个数
num_classes = target_names.shape[0]
print("数据集样本量：", num_samples)
print('特征个数：', num_features)
print('类别数量：', num_classes)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=43)
print('训练集样本量：', len(X_train))
print('测试集样本量：', len(X_test))

# PCA特征降维
num_components = 80
print('提取最前的{}个特征脸'.format(num_components))

pca = PCA(n_components=num_components, svd_solver='randomized', whiten=True).fit(X_train)
eigenfaces = pca.components_.reshape((num_components, h, w))  # 特征脸
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# 使用grid search和cross validation寻找SVM的最优参数组合
param_grid = {'C': [1, 10, 100, 500, 1e3, 5e3, 1e4, 5e4, 1e5],  # 备选参数
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)
print("最优的SVM参数组合:")
print(clf.best_estimator_)
print(clf.best_estimator_.n_support_)

# SVM训练结果
y_pred_train = clf.predict(X_train_pca)
print(classification_report(y_train, y_pred_train, target_names=target_names))

# SVM测试结果
y_pred = clf.predict(X_test_pca)
print(classification_report(y_test, y_pred, target_names=target_names))
print('混淆矩阵：', confusion_matrix(y_test, y_pred, labels=range(num_classes)))


# 可视化函数
def visualization(images, titles, h, w, n_row=3, n_col=6):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        # Show the feature face
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


def naming(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predict: %s\n label:   %s' % (pred_name, true_name)


# 混淆矩阵可视化
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    plt.axis("equal")

    ax = plt.gca()
    left, right = plt.xlim()
    ax.spines['left'].set_position(('data', left))
    ax.spines['right'].set_position(('data', right))
    for edge_i in ['top', 'bottom', 'right', 'left']:
        ax.spines[edge_i].set_edgecolor("white")

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        num = '{:.2f}'.format(cm[i, j]) if normalize else int(cm[i, j])
        plt.text(j, i, num,
                 verticalalignment='center',
                 horizontalalignment="center",
                 color="white" if num > thresh else "black")

    plt.ylabel('Label')
    plt.xlabel('Predict')

    plt.tight_layout()
    plt.show()

# 实验结果展示
prediction_titles = [naming(y_pred, y_test, target_names, i)
                     for i in range(y_pred.shape[0])]
visualization(X_test, prediction_titles, h, w, 3, 6)
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
visualization(eigenfaces, eigenface_titles, h, w, 3, 6)
plt.show()

# 混淆矩阵展示
trans_mat = confusion_matrix(y_test, y_pred, labels=range(num_classes))
label = ['Ariel', 'Colin', 'Donald', 'George', 'Gerhard', 'Hugo', 'Tony']
plot_confusion_matrix(trans_mat, label)