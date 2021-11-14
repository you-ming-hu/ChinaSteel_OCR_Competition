# 模型訓練流程簡述

辨識模型拆分成三個子模型依序為:

1. 分割圖片文字框(segment)
2. 辨識並翻正顛倒圖片(overturn)
3. 辨識分割且正確朝向的文字圖片內容(recognition)

# 資料管線簡述

依比賽階段切分三個主要資料夾:

1. train: 對應官方提供的public train以及public validation，存放訓練以上三個子模型所需的資料，並依據需求擺放資料(以下資料夾介紹部分詳述)
2. test: 對應官方提供的public test，整合利用train data訓練好的各模型，模擬正式上線時可能會遇到的問題，並提交答案，確認是否提交符合官方要求的格式、評估模型正確率。
3. predict: 對應官方提供的private test，依據test所使用的架構進行最終提交。

# 資料夾介紹

## 1. trian

**有圖片字串的標準答案，public_training_data.csv與public_testing_data.csv，前者有BBOX後者無。**

1. **official**

   訓練能夠偵測文字框的模型。由於只有train_table有給BBOX，故訓練時將train切分成train及validation檢查loss的變化。將validation當作test使用，因其無BBOX無法計算loss，僅進行視覺化輸出圖片、標準答案與模型預測答案，供人工快速檢視模型成果及改進方向。

   1. train: 來自官方提供 public_training_data.zip\public_training_data\public_training_data，含12067張圖片，圖片名稱為亂數編碼ID。
   2. validation: 來自官方提供 public_training_data.zip\public_training_data\public_testing_data，含6037張圖片，圖片名稱為亂數編碼ID。
   3. train_table.csv: 來自官方提供 標記與資料說明.zip\Training Label\public_training_data.csv，train data的標準答案，每個圖片ID都對應標準答案文字串，及該文字串的BBOX。
   4. test_table.csv: 來自官方提供 標記與資料說明.zip\Training Label\public_testing_data.csv，validation data的標準答案，每個圖片ID都對應標準答案文字串，但是沒有該文字串的BBOX。

2. **segment**

   訓練能夠偵測文字方向的模型，人工標記，將切割後的圖片分為三類: 1.明顯分割錯誤的圖片(bad_image) 2.正確方向的圖片(normal) 3.翻轉180度的圖片(overturn)，以normal、overturn切分出train及validation進行訓練，並以資料夾名稱作為label。以test data進行視覺化，供人工快速檢視模型成果及改進方向。

   1. image: 透過偵測文字框的模型，將train及validation分割好後匯合放置於此處，一共18104張圖片(12067+6037)。
   2. compare: 將分割前圖片標示BBOX後建立新備份至此，以利檢視模型成效及驗證。
   3. record: 紀錄每張圖片的分割資訊，依來源分為train.csv及validation.csv，讀取csv以快速找出不合理的BBOX長寬比，例如多字元的字串不太可能有1比1的長寬比，如果有很可能是錯誤的BBOX。
   4. image_manually_classified_by_overturn: 部分image中圖片字串180度翻轉，複製image中所有圖片，進行人工分類。
      1. bad_image: 明顯分割錯誤的圖片36張(順便檢視偵測文字框的模型表現)
      2. normal: 正確方向的圖片10192張
      3. overturn: 翻轉180度的圖片2412張
      4. 尚未分類的圖片5467張

3. **overturn**

   訓練OCR模型，將train與validation圖片均進行分割及轉正處理後合併，並以cleaned_merge_table.csv作為標準答案，最後再自行分裂出train及validation計算loss，利用test data進行視覺化，供人工快速檢視模型成果及改進方向。

   1. image: 利用偵測文字方向的模型將segment\image中圖片轉為正確方向存放於此，一共18104張圖片。
   2. cleaned_merge_table.csv: 合併train_table.csv與test_table.csv並校正部分辨識錯誤的字串答案。

## 2. test

**只有圖片沒有答案**

1. **official**: 來自官方提供 public_testing_data.zip\public_testing_data 含6000張圖片，圖片名稱為亂數編碼ID。
2. **segment**
   1. image: 透過偵測文字框的模型將official的圖片切割至此。
   2. compare: 與train相同，檢查用
   3. record.csv: 與train相同，檢查用
3. **overturn**: 利用偵測文字方向的模型將segment\image中圖片轉為正確方向存放於此。
4. **submissions**: 利用OCR模型辨識overturn中圖片並輸出答案。

## 3. predict

**只有圖片沒有答案**

1. **official**: 來自官方提供 private_data_v2.zip\private_data_v2 含9915張圖片，圖片名稱為亂數編碼ID。
2. **segment**
   1. image:  同test。
   2. compare: 同test。
   3. record.csv:  同test。
3. **overturn**:  同test。
4. **submissions**:  同test。

