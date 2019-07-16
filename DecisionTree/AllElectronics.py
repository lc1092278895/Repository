from sklearn.feature_extraction import DictVectorizer
import csv
import pydotplus
from sklearn import tree
from sklearn import preprocessing
from sklearn.externals.six import StringIO

# Read in the csv file and put features into list of dict and list of class label
allElectronicsData = open(r'./AllElectronics.csv', 'r')
reader = csv.reader(allElectronicsData)
headers = next(reader)  #读取文件第一行
print(headers)

featureList = []
labelList = []

for row in reader:
    labelList.append(row[len(row)-1]) #取最后一列
    rowDict = {}
    for i in range(1, len(row)-1):
        rowDict[headers[i]] = row[i]  ##将数据保存为字典，不含最后一列  ##  rowDict一开始是空的，rowDict[headers[i]]=row[i]是说rowDict字典的增加一个key ：headers[i]， 然后数值是row[i]
    featureList.append(rowDict)

print(featureList)

# Vetorize features
# 非数值类型变为数值矩阵
vec = DictVectorizer()
dummyX = vec.fit_transform(featureList) .toarray()

print("dummyX: \n" + str(dummyX))
print(vec.get_feature_names())

print("labelList: " + str(labelList))

# vectorize class labels
# yes和no转化为0和1
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(labelList)
print("dummyY:\n " + str(dummyY))

# Using decision tree for classification
# 使用决策树进行分类clf = tree.DecisionTreeClassifier()
# clf = tree.DecisionTreeClassifier()
clf = tree.DecisionTreeClassifier(criterion='entropy',random_state=0)
clf = clf.fit(dummyX, dummyY)
print("clf: " + str(clf))


# 生成决策树pdf图
# Visualize model
with open("allElectronicInformationGainOri.dot", 'w') as f:
    f = tree.export_graphviz(clf, feature_names=vec.get_feature_names(), out_file=f)

#验证模型
#拿第一行数据进行修改，测试模型结果
oneRowX = dummyX[0, :]
print("oneRowX: " + str(oneRowX))

newRowX = oneRowX
newRowX[0] = 1
newRowX[2] = 0
print("newRowX: " + str(newRowX))
predictedY = clf.predict(newRowX.reshape(1, -1)) #需要将数据reshape(1, -1)处理
#predictedY = clf.predict([newRowX])# 用训练好的分类器去预测
print("predictedY: " + str(predictedY))


