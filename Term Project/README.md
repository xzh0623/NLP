# 簡歷篩選
## 專案介紹
在招聘過程中，企業通常會收到大量申請。簡歷篩選能幫助快速縮小候選人範圍，評估應聘者的經驗、技能和資格是否符合職位要求，確保進入下一階段的應聘者具備基本條件。

自動化的簡歷篩選工具（如Applicant Tracking Systems, ATS）利用關鍵字搜索自動篩選出符合條件的簡歷，提高篩選效率並減少人工錯誤。

本專案將 透過自製簡歷篩選系統來快速篩選出符合各大企業所需人才條件。
## 專案分析過程
### 1. 資料收集
從Kaggle官網下載2種資料集。

第1種資料集包含求職者的簡歷文本以及期望職務，共2482位求職者。  
https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset
* ID：簡歷編號
* Resume_str：簡歷文本的字符串格式
* Resume_html：網頁抓取時的簡歷數據的HTML格式
* Category：期望職務，包含人力資源、設計師、信息技術、教師、律師、業務發展、醫療保健、健身、農業、商務流程外包、銷售、顧問、數字媒體、汽車、廚師、金融、服裝、工程、會計、建築、公關、銀行、藝術、航空

第2種資料集為各企業期望的職務條件，共853家企業。  
https://www.kaggle.com/datasets/pranavvenugo/resume-and-job-description?select=training_data.csv
* company_name：企業名稱
* job_description：職位描述與條件要求文本
* position_title：招聘職位，包含銷售專員、網頁設計師、高級UI設計師、CEO、教練等
* description_length：職位描述長度
* model_response：職位描述分類，包含核心職責、所需技能、學歷要求、經驗水準、首選資格、薪資福利
### 2. 資料預處理
對資料集(求職者的簡歷文本和企業期望的職務條件)進行文本清洗、分詞、去除停用詞、詞幹化或詞形還原等預處理操作 
### 3. 特徵提取
使用TF-IDF 向量、ContVectorizer、詞嵌入（如Word2Vec、GloVe）等方法將文本轉換為數值特徵表示
* Word2Vec模型使用 職位描述分類、招聘職位、簡歷文本訓練，並供後續分析使用
### 4. 資料分析
使用已標記的求職者簡歷文本和企業期望職務條件進行分析。
* 方法1：聚類算法
  1. 使用K-means進行履歷的聚類
  2. 觀察每個cluster的特徵 (取在cluster中出現最高頻率的前10個關鍵字)
  3. 一筆職位描述，與每個cluster計算相似度，找到與之最相似的cluster
  4. 計算每種方法「職位描述與最相似的cluster」的平均相似度
  5. 查看最相似的cluster中，職位描述與履歷相似度最高的前三名

* 方法2：相似度計算  
  使用GloVe模型將 企業職位﻿描述job_description 和 簡﻿歷文本Resume_str 轉換為向量形式後，使用cosine similarity計算兩者間的相似度  
  最高：0.94～0.91 最低：0.58～0.62
## 專案結果
列出與各企業職位描述最相符的前3位求職者
(共853筆，僅取前4筆)
![image](https://github.com/xzh0623/NLP/assets/110615484/16fd7d24-4ba4-4ed4-9923-2a4e7299666a)
各企業招收職位與求職者期望職位
(共853筆，僅取前4筆)
![image](https://github.com/xzh0623/NLP/assets/110615484/e721b9d5-261a-4281-bde9-25cf5de56de7)
## 結果探討
由上述結果可看出求職者履歷內容與各企業所需職位條件相符，但與求職者期望職位不符
* 解決方案：
  1. 多方法融合
將 TF-IDF、CountVectorizer、Word2Vec 和 GloVe 的相似度結果進行融合，計算綜合相似度。
  2. 嘗試調整k-means的聚類數量，觀察結果表現
  3. 嘗試其他聚類及相似度計算方法

