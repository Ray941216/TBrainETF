
# coding=utf8

# # ETF18

# # 準備資料

# In[1]:


# get_ipython().system('ls DataSet/')


# In[2]:


import talib as ta
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import pandas as pd
from urllib import request
import requests
from time import sleep
import chardet
import sys

def get_file_list(root, ftype = ".csv"):
    import os
    import sys
    FileList = []
    filename = []
    for dirPath, dirNames, fileNames in os.walk(root):
        for f in fileNames:
            if f.find(ftype) > -1:
                FileList.append(os.path.join(dirPath, f))
                filename.append(f.replace(ftype, ""))
    if len(filename) > 0:
        a = zip(FileList, filename)
        a = sorted(a, key = lambda t : t[1])
        FileList, filename = zip(*a)
    return FileList, filename

def crawl3big(date_list):
    if len(date_list) > 0:
        with requests.Session() as s:
            s.proxies = {
              'http': 'http://54.153.86.193:8080',
              'https': 'http://153.149.171.53:3128',
            }
            s.headers.update({'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/65.0.3325.181 Safari/537.36'})

            for d in date_list:
                sys.stdout.write("\rNow is fetching the data of "+ str(d))
                resp = s.get("http://www.twse.com.tw/fund/T86?response=csv&date="+str(d)+"&selectType=ALL")
                lines = []
                for l in resp.content.decode('big5hkscs').split("\n")[1:]:
                    lines.append(l)

                with open("./DataSet/"+str(d)+".csv", "w", encoding='big5hkscs') as f:
                    f.writelines(lines)
                sleep(5)


# In[3]:


def squash(x):
    lengths2 = np.sum(np.power(x, 2))
    lengths = np.sqrt(lengths2)
    x = x * np.divide(lengths, (1 + lengths2))
    return x


sys.stdout.write("\r1...")

# In[4]:


etf18 = pd.read_csv("./DataSet/tetfp.csv", encoding="big5hkscs", low_memory=False)
etf18 = etf18.rename(columns={'中文簡稱':'證券名稱'})
etf18 = etf18.dropna(axis=1, how='all')
etf18 = etf18.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
etf18 = etf18.drop_duplicates()
etf18 = etf18.reset_index()
code_uq = list(etf18["代碼"].unique())

# In[5]:


# etf18


# In[6]:


code = etf18.pop("代碼").values
op = etf18.pop("開盤價(元)").values
hi = etf18.pop("最高價(元)").values
lo = etf18.pop("最低價(元)").values
cl = etf18.pop("收盤價(元)").values
vo = etf18.pop("成交張數(張)").values
# names = list(np.unique(etf18["證券名稱"].values))


# In[7]:





# In[8]:


# closeDF


# In[9]:


for i in range(vo.shape[0]):
    vo[i] = float(str(vo[i]).replace(",", ""))
    cl[i] = float(str(cl[i]).replace(",", ""))
    hi[i] = float(str(hi[i]).replace(",", ""))
    lo[i] = float(str(lo[i]).replace(",", ""))
    op[i] = float(str(op[i]).replace(",", ""))


# In[10]:
closeDF = pd.DataFrame(np.array(cl).T, columns=["close"])

op_pkg = []
hi_pkg = []
lo_pkg = []
cl_pkg = []
vo_pkg = []
date_list = list(np.unique(etf18["日期"].values))


try:
    finalETF18 = pd.read_csv("../TBrain/DataSet/final18_"+str(date_list[-1])+".csv")
except:
    no_data_prepared = True
else:
    no_data_prepared = False
    del etf18

# In[11]:

if no_data_prepared:
    idx = 0
    for i in range(1, len(code)):
        if code[idx] != code[i]:
            op_pkg.append(list(op[idx:i]))
            hi_pkg.append(list(hi[idx:i]))
            lo_pkg.append(list(lo[idx:i]))
            cl_pkg.append(list(cl[idx:i]))
            vo_pkg.append(list(vo[idx:i]))
            idx = i


    op_pkg.append(list(op[idx:len(code)]))
    hi_pkg.append(list(hi[idx:len(code)]))
    lo_pkg.append(list(lo[idx:len(code)]))
    cl_pkg.append(list(cl[idx:len(code)]))
    vo_pkg.append(list(vo[idx:len(code)]))


    # In[12]:


    cl_Box = [[],[],[],[],[]]
    for idx, c in enumerate(cl_pkg):
        for i in range(0, 5):
            cl_Box[i] += (list(c[i+1:]) + [np.nan for x in range(min(i+1, len(c)))])

    clBoxDF = pd.DataFrame(np.array(cl_Box).T, columns=['cl1', 'cl2', 'cl3', 'cl4', 'cl5'])


    # In[13]:


    closeDF = pd.concat([closeDF, clBoxDF], axis=1)


    # In[14]:


    #closeDF


    sys.stdout.write("\r2....")

    # In[15]:


    sys.stdout.write("\r\t\t\t\t\t\r3............")
    lst = []
    flist, ftag = get_file_list("../TBrain/DataSet/")
    for d in date_list:
        if str(d) not in ftag:
            lst.append(d)
    if len(lst) >0:
        crawl3big(lst)


    # In[16]:


    three = pd.read_csv("./DataSet/"+str(date_list[0])+".csv", encoding='big5hkscs')
    three.insert(0,'日期',[date_list[0] for i in three.index])
    try:
        discrd = three.pop("證券代號")
    except:
        print("Error of 證券代號 on "+str(date_list[0]))

    three = three.dropna(axis=1, how='all')
    three = three.dropna(axis=0,how='any')

    for d in date_list[1:]:
        try:
            a = pd.read_csv("./DataSet/"+str(d)+".csv", encoding='big5hkscs')
        except:
            print("Error of OPEN-FILE on "+str(d))
            break

        a.insert(0,'日期',[d for i in a.index])
        try:
            discrd = a.pop("證券代號")
        except:
            print("Error of 證券代號 on "+str(d))
            break

        a = a.dropna(axis=1, how='all')
        a = a.dropna(axis=0,how='any')
        pd.concat([three, a], axis=0, join='outer', join_axes=None, ignore_index=False,
              keys=None, levels=None, names=None, verify_integrity=False,
              copy=True)
        three = three.append(a, ignore_index=True)

    three = three.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    three = three.fillna(0)


    # In[17]:


    #three


    # In[18]:


    res = pd.merge(etf18, three, on=['日期', '證券名稱'], how='left')
    res = res.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    res = res.fillna(0)


    # In[19]:


    #res


    # In[20]:


    newVal = []
    newVal.append(res.pop("日期"))
    newVal.append(res.pop("證券名稱"))

    newDF = pd.DataFrame(np.array(newVal).T)


    # In[21]:


    threeVal = res.values
    newVal = []
    for x in threeVal:
        x2 = []
        for e in x:
            x2.append(float((str(e)).replace(",", "")) / 1000)
        newVal.append(squash(np.array(x2)))

    new3DF = pd.DataFrame(newVal)


    # In[22]:


    #new3DF


    sys.stdout.write("\r\t\t\t\t\r4........")

    # ## KDJ 指標
    #
    # ### KDJ指標的參數設置
    #
    # KDJ指標參數設置的核心原則：設置參數既需要因人而異，也要因行情而異。
    # 在一般的分析軟體中，KDJ指標的默認參數是（9，3，3）。從實戰的角度來看，由這一參數設置而成的日K線KDJ指標存在著波動頻繁，過於靈敏，無效信號較多的缺陷，也正因為如此，KDJ指標往往被投資者所忽略，認為這一指標並沒有太大的使用價值。但事實上，如果把KDJ指標的參數進行修改，可以發現這一指標對判研價格走勢仍然具有比較好的功效。
    #
    #
    # #### 1.以（6，3，3）為參數而設置的KDJ指標
    #
    # 對價格波動的敏感性得到加強，它變動的頻率非常高，適合於短線投資者尋找買賣點。一般來說，KDJ三條線在超買超賣區的每一次交叉都可能成為重要的操作時點。
    # #### 2.以（18，3，3）為參數設置而成的KDJ指標
    # 具有信號穩定而且靈敏度不低的優點，在大多數情況下都比較適用。按照這一參數設定的KDJ指標，有一條非常重要的操作原則就是：在指標處於20超賣區以下，並且指標出現底背離時，應該買進；而在80超賣區以上，指標出現頂背離時應該賣出。
    # #### 3.以（24，3，3）為參數而設定的KDJ指標
    # 在更大程度上排除了價格波動所產生的虛假信號，用它來尋找價格的中線買點或者是中線賣點是一個比較好的選擇。
    # #### 參數設置後需要注意的要點：
    # ##### 1.以參數（6，3，3）設置而成的KDJ指標
    # 由於在價格運行中會出現多次交叉，容易產生信號失真，因此需要投資者有足夠的實戰經驗，初學者一般不要輕易嘗試。
    #
    # ##### 2.以參數（18，3，3）設置而成的KDJ指標
    # 在使用中必須遵循這樣的操作原則：
    # (1)指標的交叉必須是出現在超賣區或者超買區時才是有效信號
    # (2)在底部發出交叉時，出現兩次交叉應視為良好的買進時機
    # (3)在高位出現交叉時，出現兩次交叉應視為良好的賣出時機
    #
    #
    #

    # In[23]:


    def KDJ(high, low, close, rsv = 9, k = 3, d = 3):
        K, D = ta.STOCH(np.array(high), np.array(low), np.array(close), fastk_period=rsv, slowk_period=k, slowk_matype=0, slowd_period=d, slowd_matype=0)
        return K, D, (3 * D - 2 * K)


    # In[24]:


    paras = [(9,3,3), (6,3,3), (18,3,3), (24, 3, 3), (5,21,11)]
    kdj = pd.DataFrame()
    for hi, lo, cl in zip(hi_pkg, lo_pkg, cl_pkg):
        box = []
        for p in paras:
            k, d, j = KDJ(hi, lo, cl, p[0], p[1], p[2])
            box.append(k)
            box.append(d)
            box.append(j)
        kdj = kdj.append(pd.DataFrame(data=np.array(box).T), ignore_index=True)


    # In[25]:


    #kdj


    # In[26]:


    def PriceLoc(close, high, low, period = [6, 11, 22, 43, 65, 130]):
        close = np.array(close)
        high = np.array(high)
        low = np.array(low)
        WCP = ta.WCLPRICE(high, low, close)

        sma_cl = []
        for p in period:
            sma_cl.append(ta.SMA(close, timeperiod=p))
        sma_wcl = []
        for p in period:
            sma_wcl.append(ta.SMA(WCP, timeperiod=p))

        period.reverse()
        period_r = np.array([period])

        wsma_cl = np.sum(period_r.T * sma_cl, axis=0) / sum(period)
        wsma_wcl = np.sum(period_r.T * sma_wcl, axis=0) / sum(period)

        space = []
        space.append((WCP - wsma_cl) / wsma_cl)
        space.append((WCP - wsma_wcl) / wsma_wcl)
        for i in range(len(sma_cl)):
            space.append((WCP - sma_cl[i]) / sma_cl[i])
        for i in range(len(sma_wcl)):
            space.append((WCP - sma_wcl[i]) / sma_wcl[i])

        return space


    # In[27]:


    def VolLoc(volume, period = [6, 11, 22, 43, 65, 130]):
        volume = np.array(volume)

        smv = []
        for p in period:
            smv.append(ta.SMA(volume, timeperiod=p))

        period.reverse()
        period_r = np.array([period])

        wsmv = np.sum(period_r.T * smv, axis=0) / sum(period)

        space = []
        space.append((volume - wsmv) / wsmv)
        for i in range(len(smv)):
            space.append((volume - smv[i]) / smv[i])

        return space


    # In[28]:


    def StockLoc(close, high, low, volume, period = [6, 11, 22, 43, 65]):
        return PriceLoc(close, high, low, period) + VolLoc(volume, period)


    # In[29]:


    db = []
    for i in range(len(cl_pkg)):
        db.append(StockLoc(cl_pkg[i], hi_pkg[i], lo_pkg[i], vo_pkg[i]))


    # In[30]:


    loc = pd.DataFrame()
    for d in db:
        loc = loc.append(pd.DataFrame(data=np.array(d).T), ignore_index=True)


    # In[31]:


    #loc


    # In[32]:


    def ClosePercenting(close, period = [1, 2, 3, 4, 5]):
        period.sort()
        cp = []
        for p in period:
            cp.append([])

        for i in range(len(close) - period[-1]):
            for j, p in zip(range(len(period)), period):
                cp[j].append((close[i + p] - close[i]) / close[i] * 100)

        for i in range(period[-1]):
            for j in range(len(period)):
                cp[j].append(np.nan)

        data_dict = {}
        for j in range(len(period)):
            data_dict.update({"change%_D"+str(period[j]) : cp[j]})

        out = pd.DataFrame(data=data_dict)
        return out


    # In[33]:


    change = ClosePercenting(cl_pkg[0])
    for cl in cl_pkg[1:]:
        change = change.append(ClosePercenting(cl), ignore_index=True)


    # In[34]:


    #change


    # ### Pattern Recognition Functions

    # In[35]:


    func_list = [ta.CDL2CROWS, ta.CDL3BLACKCROWS, ta.CDL3INSIDE, ta.CDL3LINESTRIKE, ta.CDL3OUTSIDE, ta.CDL3WHITESOLDIERS, ta.CDLABANDONEDBABY, ta.CDLADVANCEBLOCK, ta.CDLBELTHOLD, ta.CDLBREAKAWAY, ta.CDLCLOSINGMARUBOZU, ta.CDLCONCEALBABYSWALL, ta.CDLCOUNTERATTACK, ta.CDLDARKCLOUDCOVER, ta.CDLDOJI, ta.CDLDOJISTAR, ta.CDLDRAGONFLYDOJI, ta.CDLENGULFING, ta.CDLEVENINGDOJISTAR, ta.CDLEVENINGSTAR, ta.CDLGAPSIDESIDEWHITE, ta.CDLGRAVESTONEDOJI, ta.CDLHANGINGMAN, ta.CDLHARAMI, ta.CDLHARAMICROSS, ta.CDLHIGHWAVE, ta.CDLHIKKAKE, ta.CDLHIKKAKEMOD, ta.CDLHOMINGPIGEON, ta.CDLIDENTICAL3CROWS, ta.CDLINNECK, ta.CDLINVERTEDHAMMER, ta.CDLKICKING, ta.CDLKICKINGBYLENGTH, ta.CDLLADDERBOTTOM, ta.CDLLONGLEGGEDDOJI, ta.CDLLONGLINE, ta.CDLMARUBOZU, ta.CDLMATCHINGLOW, ta.CDLMATHOLD, ta.CDLMORNINGDOJISTAR, ta.CDLMORNINGSTAR, ta.CDLONNECK, ta.CDLPIERCING, ta.CDLRICKSHAWMAN, ta.CDLRISEFALL3METHODS, ta.CDLSEPARATINGLINES, ta.CDLSHOOTINGSTAR, ta.CDLSHORTLINE, ta.CDLSPINNINGTOP, ta.CDLSTALLEDPATTERN, ta.CDLSTICKSANDWICH, ta.CDLTAKURI, ta.CDLTASUKIGAP, ta.CDLTHRUSTING, ta.CDLTRISTAR, ta.CDLUNIQUE3RIVER, ta.CDLUPSIDEGAP2CROWS, ta.CDLXSIDEGAP3METHODS]


    # In[36]:


    ptnDF = pd.DataFrame()
    for i in range(len(code_uq)):
        a = []
        for f in func_list:
            a.append(f(np.array(op_pkg[i]), np.array(hi_pkg[i]), np.array(lo_pkg[i]), np.array(cl_pkg[i])))

        ptnDF = ptnDF.append(pd.DataFrame(np.array(a).T), ignore_index=True)

    ptnVal = ptnDF.values
    ptnValSQ = []
    for x in ptnVal:
        ptnValSQ.append(squash(x))

    ptnDFSQ = pd.DataFrame(ptnValSQ)


    # In[37]:


    #ptnDFSQ


    # In[38]:


    finalETF18 = pd.concat([etf18, change, closeDF, loc, kdj, new3DF, ptnDFSQ], axis=1)


    # In[39]:


    #finalETF18


    # In[40]:


    finalETF18.to_csv("./DataSet/final18_"+str(date_list[-1])+".csv", index=False)


sys.stdout.write("\r\t\t\t\t\r5......")

# In[41]:


finalETF18 = finalETF18.dropna() # 去除空資訊
finalETF18 = finalETF18.drop_duplicates()
discard = finalETF18.pop("日期")# 移除日期
tags = finalETF18.pop("證券名稱").values # 取得標籤
dataVal = finalETF18.values


# In[42]:


dataValSplit = []

idx = 0
for i in range(1, len(tags)):
    if tags[idx] != tags[i]:
        dataValSplit.append(list(dataVal[idx:i]))
        idx = i

dataValSplit.append(list(dataVal[idx:len(dataVal) - 1]))


# In[43]:


frames = []
for d in dataValSplit:
    frames.append(pd.DataFrame(d))


# In[44]:


#frames[0]


# In[45]:


Y_box = []
for f in frames:
    tmp = []
    for i in range(0, 11):
        tmp.append(f.pop(i).values)
    Y_box.append(tmp)


# In[46]:


Y_frames = []
for y in Y_box:
    Y_frames.append(pd.DataFrame(np.array(y).T, columns=['d1', 'd2', 'd3', 'd4', 'd5', 'close', 'cl1', 'cl2', 'cl3', 'cl4', 'cl5']))


# In[47]:


#Y_frames[0]


# In[48]:


X_box = []
for f in frames:
    X_box.append(f.values)


# In[49]:


def train_test_split(data_box, target_box, train_rate=.9, test_rate=.1):
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    if (train_rate + test_rate != 1):
        test_rate = 1 - train_rate

    for i in range(len(data_box)):
        tot = len(data_box[i]) # 總長度
        tra = int(np.ceil(tot * train_rate) - 1)
        x_train.append(list(data_box[i][:tra]))
        x_test.append(list(data_box[i][tra:]))

        tmp_ytr = []
        tmp_yte = []
        for j in range(len(target_box[i])):
            tmp_ytr.append(list(target_box[i][j][:tra]))
            tmp_yte.append(list(target_box[i][j][tra:]))
        y_train.append(tmp_ytr)
        y_test.append(tmp_yte)

    return x_train, y_train, x_test, y_test


# In[50]:


x_train, y_train, x_test, y_test = train_test_split(X_box, Y_box, 0.9)

x_50tr, y_50tr, x_50te, y_50te = train_test_split(X_box, Y_box, 0.5)
x_ftr, y_ftr, x_fte, y_fte = train_test_split(x_50te, y_50te, 0.8)


# In[51]:


#len(y_train[0])


# In[52]:


import torch
from torch.autograd import Variable as var
from torch import nn, cuda
from torch.utils.data import Dataset, DataLoader, sampler


# In[53]:


class DSet(Dataset):
    def __init__(self, x, y):
        super(DSet, self).__init__()
        self.base_x = x
        self.base_y = y
    def __getitem__(self, index):
        while index < 0:
            index += self.__len__()
        while index >= self.__len__():
            index -= self.__len__()

        for i in range(len(self.base_x)):
            if index > len(self.base_x[i]) - 1 - 8:
                index -= (len(self.base_x[i]) - 1 - 8)
            else:
                # y will output ['d1', 'd2', 'd3', 'd4', 'd5', 'close', 'cl1', 'cl2', 'cl3', 'cl4', 'cl5']
                y = np.array([self.base_y[i][x][index + 8] for x in range(len(self.base_y[i]))])
                return {"X":(torch.FloatTensor([self.base_x[i][index : index + 9]])), "Y":y, "close":(torch.FloatTensor([self.base_y[i][5][index + 8]]))}

    def __len__(self):
        tot = 0
        for x in self.base_x:
            if (len(x) - 9) >= 0:
                tot += (len(x) - 9)
            else:
                tot += 0
        return tot + 1


# In[54]:


class DSet2(Dataset): # for fine tune
    def __init__(self, x, y):
        super(DSet2, self).__init__()
        self.base_x = x
        self.base_y = y
    def __getitem__(self, index):
        while index < 0:
            index += self.__len__()
        while index >= self.__len__():
            index -= self.__len__()


        # y will output ['d1', 'd2', 'd3', 'd4', 'd5', 'close', 'cl1', 'cl2', 'cl3', 'cl4', 'cl5']
        y = np.array([self.base_y[x][index + 8] for x in range(len(self.base_y))])
        return {"X":(torch.FloatTensor([self.base_x[index : index + 9]])), "Y":y, "close":(torch.FloatTensor([self.base_y[5][index + 8]]))}


    def __len__(self):
        tot = 0
        tot += (len(self.base_x) - 9)
        if tot < 0:
            tot = -1
        return tot + 1


# In[55]:


train = DSet(x_train, y_train)
valid = DSet(x_test, y_test)

finetune_tr = []
for i in range(len(x_ftr)):
    finetune_tr.append(DSet2(x_ftr[i], y_ftr[i]))

finetune_te = []
for i in range(len(x_fte)):
    finetune_te.append(DSet2(x_fte[i], y_fte[i]))


# In[56]:


#train[0]


# # Model Building

# In[57]:


log_softmax = lambda x : np.round(np.log(np.exp(x) / np.sum(np.exp(x))), 4)
def Build_Y(cp_1D, classes = [x for x in np.arange(-9.5, 10, 1)]):
    out = []
    for x in cp_1D:
        tmp = []
        for c in classes:
            tmp.append(-(abs(x - c)))
        out.append(log_softmax(tmp))
    return out


# In[58]:


class SubNet2(nn.Module):
    def __init__(self):
        super(SubNet2, self).__init__()
        self.lstm = nn.LSTM(
            input_size = 666,
            hidden_size = 40,
            num_layers = 2,
            bidirectional = True
        )
        self.h = None
        self.fc = nn.Linear(80, 20)
        self.mix = nn.Linear(21, 1)

        # self.myShift = lambda n : (n + (torch.mean(torch.abs(torch.min(n)) + torch.abs(torch.max(n))) + 1)) / torch.sum((n + (torch.mean(torch.abs(torch.min(n)) + torch.abs(torch.max(n))) + 1))) if torch.equal(torch.min(n).cpu() < 0, var(torch.ByteTensor([1]))) or torch.equal(torch.max(n).cpu() > 1, var(torch.ByteTensor([1]))) else n

    def encode(self, semi_code):
        code, self.h = self.lstm(semi_code, self.h)
        code = self.fc(code)
        return code

    def decode(self, code, val):
        ans = self.mix(torch.cat((torch.exp(code.view(1, -1)), val.view(1, -1)), 1))
        return ans

    def forward(self, x, val):
        code = self.encode(x)
        code = nn.functional.log_softmax(code, 2)
        out = self.decode(code, val)
        return code, out

class TopNet2(nn.Module):
    def __init__(self):
        super(TopNet2, self).__init__()
        # (1, 1, 9, 114)
        self.explode = nn.Conv2d(1, 6, 7)
        # (1, 5, 3, 108)
        # (1, 1, 18, 18, 18)
        #self.compress = nn.Conv3d(1, 3, 13)
        # (1, 3, 6, 6, 6)
        # (1, 1, 18, 36)
        fib = lambda n : n if n < 2 else fib(n - 1) + fib(n - 2)
        self.diggers = [nn.Conv2d(1, 1, fib(i + 3)) for i in range(5)]
        for m in range(len(self.diggers)):
            self.add_module('digger-'+str(m + 1), self.diggers[m])
        # (1, 1, 6658)
        self.filter= [nn.Linear(6658, 666)for i in range(5)]
        for m in range(len(self.filter)):
            self.add_module('filter-'+str(m + 1), self.filter[m])

        self.predictor = [nn.DataParallel(SubNet2()) for i in range(5)]
        for m in range(len(self.predictor)):
            self.add_module('predictor-'+str(m + 1), self.predictor[m])

    def forward(self, x, val):
        x_exploded = self.explode(x.view(1, 1, 9, -1)).view(1, 1, 18, -1)
        # x_compressed =self.compress(x_exploded).view(1, 1, 18, -1)
        x_gold = self.diggers[0](x_exploded).view(1, 1, -1)
        for m in range(1, len(self.diggers)):
            x_gold = torch.cat((x_gold, self.diggers[m](x_exploded).view(1, 1, -1)), 2)

        code_list = []
        ans_list = []

        for m in range(5):
            code, out = self.predictor[m](self.filter[m](x_gold), val)

            code_list.append(code)
            ans_list.append(out)

        updn = []
        updn.append(code_list[0])
        for i_c in range(1, 5):
            updn.append(nn.functional.log_softmax(torch.exp(code_list[i_c]) - torch.exp(code_list[i_c - 1]), dim=2))

        return code_list, ans_list, updn


# In[60]:

class SubNet3(nn.Module):
    def __init__(self):
        super(SubNet3, self).__init__()
        self.lstm = nn.LSTM(
            input_size = 1752,
            hidden_size = 200,
            num_layers = 1,
            bidirectional = True
        )
        self.h = None
        self.fc = nn.Linear(400, 20)
        self.mix = nn.Linear(20, 1)

        # self.myShift = lambda n : (n + (torch.mean(torch.abs(torch.min(n)) + torch.abs(torch.max(n))) + 1)) / torch.sum((n + (torch.mean(torch.abs(torch.min(n)) + torch.abs(torch.max(n))) + 1))) if torch.equal(torch.min(n).cpu() < 0, var(torch.ByteTensor([1]))) or torch.equal(torch.max(n).cpu() > 1, var(torch.ByteTensor([1]))) else n

    def encode(self, semi_code):
        code, self.h = self.lstm(semi_code, self.h)
        code = self.fc(code)
        return code

    def decode(self, code, val):
        frac = self.mix(torch.exp(code.view(1, -1)))
        ans = torch.mul(val.view(1, -1), frac)
        return ans

    def forward(self, x, val):
        code = self.encode(x)
        code = nn.functional.log_softmax(code, 2)
        out = self.decode(code, val)
        return code, out

class TopNet3(nn.Module):
    def __init__(self):
        super(TopNet3, self).__init__()
        fib = lambda n : n if n < 2 else fib(n - 1) + fib(n - 2)
        # (1, 1, 9, 114)
        self.explode = [nn.Conv2d(1, 1, fib(i + 3)) for i in range(4)]
        for m in range(len(self.explode)):
            self.add_module('explode-'+str(m + 1), self.explode[m])
        # [1, 1, 74, 47]
        self.diggers = [nn.Conv2d(1, 1, fib(i + 3)) for i in range(7)]
        for m in range(len(self.diggers)):
            self.add_module('digger-'+str(m + 1), self.diggers[m])
        # [1, 1, 17516]
        self.filter= [nn.Linear(17516, 1752)for i in range(5)]
        for m in range(len(self.filter)):
            self.add_module('filter-'+str(m + 1), self.filter[m])

        self.predictor = [nn.DataParallel(SubNet3()) for i in range(5)]
        for m in range(len(self.predictor)):
            self.add_module('predictor-'+str(m + 1), self.predictor[m])

    def forward(self, x, val):
        x_exploded = x.view(1, 1, -1)
        for m in range(len(self.explode)):
            x_exploded = torch.cat((x_exploded, self.explode[m](x.view(1, 1, 9, -1)).view(1, 1, -1)), 2)
        # x_compressed =self.compress(x_exploded).view(1, 1, 18, -1)
        x_exploded = x_exploded.view(1, 1, 74, -1)
        x_gold = x.view(1, 1, -1)
        for m in range(len(self.diggers)):
            x_gold = torch.cat((x_gold, self.diggers[m](x_exploded).view(1, 1, -1)), 2)

        code_list = []
        ans_list = []

        for m in range(5):
            code, out = self.predictor[m](self.filter[m](x_gold), val)

            code_list.append(code)
            ans_list.append(out)

        updn = []
        updn.append(torch.sign(ans_list[0].view(1, -1) - val.view(1, -1)))
        for i_c in range(1, 5):
            updn.append(torch.sign(ans_list[i_c].view(1, -1) - ans_list[i_c - 1].view(1, -1)))

        return code_list, ans_list, updn


# MTL = TopNet(114, 20, 5).cuda()
# MTL = TopNet2().cuda()
MTL = TopNet3().cuda()
try:
    MTL.load_state_dict(torch.load("./MTL3_base.pth"))
except:
    try:
        MTL.load_state_dict(torch.load("./MTL3_base.pth"))
    except:
        print("NO pre-trained model.")
    else:
        print("Last State has load!")
else:
    print("Last State has load!")


# In[61]:


#MTL


# # Training



# In[70]:


Learning_Rate = 1e-4
EPOCH = 50
BATACH_SIZE = 22


# In[72]:


train_dl = DataLoader(dataset=train, batch_size=BATACH_SIZE, num_workers=6, sampler=sampler.SequentialSampler(train), pin_memory=True)
#train_dl = DataLoader(dataset=train, batch_size=BATACH_SIZE, num_workers=6, pin_memory=True, shuffle=True)
valid_dl = DataLoader(dataset=valid, batch_size=5, num_workers=6, sampler=sampler.SequentialSampler(valid), pin_memory=True)

finetune_tr_dl = []
finetune_te_dl = []
for i_ftr, ftr in enumerate(finetune_tr):
    finetune_tr_dl.append(DataLoader(dataset=ftr, batch_size=20, num_workers=6, sampler=sampler.SequentialSampler(ftr), pin_memory=True))
for i_fte, fte in enumerate(finetune_te):
    finetune_te_dl.append(DataLoader(dataset=fte, batch_size=5, num_workers=6, sampler=sampler.SequentialSampler(fte), pin_memory=True))

# In[74]:


for p in MTL.parameters():
    p.requires_grad = True
op = torch.optim.Adam(MTL.parameters(), lr=Learning_Rate)
lf1 = nn.KLDivLoss()
lf2 = nn.MSELoss()
lf3 = nn.SmoothL1Loss()
lf4 = lambda a, b : 1 if torch.equal(a.view(1, -1), b.view(1, -1)) else 0


# Pre-train the decoder
for p in MTL.parameters():
    p.requires_grad = False
op2mix = []
for m in MTL.predictor:
    for p in m.module.parameters():
        p.requires_grad = True
    op2mix.append(torch.optim.Adam(m.module.parameters(), lr=1e-3))
    for p in m.module.parameters():
        p.requires_grad = False
    for p in m.module.mix.parameters():
        p.requires_grad = True

print("Training Decoder...")
for epo in range(0):
    epo_loss = []
    for i_batch, sample_batched in enumerate(train_dl):
        y_code_box = [Build_Y(np.array(sample_batched['Y']).T[m]) for m in range(5)]
        for mini_batch in range(sample_batched['X'].size()[0]):
            val = var(sample_batched['close'][mini_batch]).cuda()
            for i, m in enumerate(MTL.predictor):
                y1 = var(torch.FloatTensor(y_code_box[i][mini_batch])).cuda()
                ans = m.module.decode(y1, val)

                y2 = var(torch.FloatTensor([[sample_batched['Y'][mini_batch][i + 6]]])).cuda()
                op2mix[i].zero_grad()
                loss = lf3(ans, y2)
                loss.backward()

                epo_loss.append(loss.view(-1).cpu().data.numpy()[0])
                if np.isnan(epo_loss[-1]) or np.isinf(epo_loss[-1]):
                    print(i_batch, mini_batch)
                    print("\nCODE = ")
                    print(y1)
                    print("\nVAL = ")
                    print(val)
                    print("\nANS = ")
                    print(ans)
                    print("\nTrue = ")
                    print(y2)
                    # torch.save(MTL.state_dict(), "./MTL3_before_nan_B{0}.pth".format(i_batch))
                    input("Windows: Ctrl_Z+Return")
                    break
                else:
                    op2mix[i].step()
                    """sys.stdout.write("\rEpoch#{0}-{1}-{2}, Loss = {3}".format(epo, i_batch, i, epo_loss[-1]))
                    if epo_loss[-1] >= 1:
                        sys.stdout.write("\n")"""
        if (i_batch + 1) % 100 == 0:
            print("100 Batches Trained!, Avg Loss = {0}".format(np.average(epo_loss[-100:])))
            #torch.save(MTL.state_dict(), "./MTL3_base_E{0}_B{1}.pth".format(epo, i_batch + 1))
    if np.average(epo_loss) <= 1e-4:
        print("\n End at the {0} epoch".format(epo+1))
        break
    else:
        torch.save(MTL.state_dict(), "./MTL3_base.pth")

for p in MTL.parameters():
    p.requires_grad = True
torch.save(MTL.state_dict(), "./MTL3_base.pth")

op_en = torch.optim.Adam(MTL.parameters(), lr=1e-4, eps=1e-4)
for m in MTL.predictor:
    for p in m.module.mix.parameters():
        p.requires_grad = False

print("Training Encoder...")
for epo in range(0):
    print("Epoch#", epo+1)
    epo_loss = []
    for i_batch, sample_batched in enumerate(train_dl):
        y_code_box = [Build_Y(np.array(sample_batched['Y']).T[m]) for m in range(5)]
        for mini_batch in range(sample_batched['X'].size()[0]):
            x = var(sample_batched['X'][mini_batch]).cuda()
            val = var(sample_batched['close'][mini_batch]).cuda()
            code_list, ans_list, updn = MTL(x, val)

            op_en.zero_grad()
            y1 = var(torch.FloatTensor(y_code_box[0][mini_batch])).cuda()
            loss = lf1(code_list[0].view(1, -1), y1) + lf2(code_list[0].view(1, -1), y1)
            for i in range(1, len(code_list)):
                y1 = var(torch.FloatTensor(y_code_box[i][mini_batch])).cuda()

                loss += lf1(code_list[i].view(1, -1), y1) + lf2(code_list[i].view(1, -1), y1)
            loss.backward(retain_graph=True)
            epo_loss.append(loss.view(-1).cpu().data.numpy()[0])
            if np.isnan(epo_loss[-1]) or np.isinf(epo_loss[-1]):
                print("\nCODE = ")
                print(code_list)
                print("\nVAL = ")
                print(val)
                print("\nANS = ")
                print(ans_list)
                torch.save(MTL.state_dict(), "./MTL3_before_nan_B{0}.pth".format(i_batch))
                break
            else:
                op_en.step()
                """sys.stdout.write("\rEpoch#{0}, Loss = {1}".format(epo, epo_loss[-1]))
                if epo_loss[-1] >= 1:
                    sys.stdout.write("\n")"""
        if (i_batch + 1) % 100 == 0:
            print("100 Batches Trained!, Avg Loss = {0}".format(np.average(epo_loss[-100:])))
            #torch.save(MTL.state_dict(), "./MTL3_base_E{0}_B{1}.pth".format(epo, i_batch + 1))
    if np.average(epo_loss) <= 10:
        print("\n End at the {0} epoch".format(epo+1))
        break
    else:
        torch.save(MTL.state_dict(), "./MTL3_base.pth")

for p in MTL.parameters():
    p.requires_grad = True
torch.save(MTL.state_dict(), "./MTL3_base.pth")

# In[75]:
_100 = var(torch.FloatTensor([100])).cuda()
Result = []
lrs = [(1e-2 / 2), 1e-3]
w = [0.1, 0.15, 0.2, 0.25, 0.3]
opctrl = {}
for i in range(0, 5):
    if i not in [4, 3, 2]:
        opctrl.update({str(i):0})
    else:
        opctrl.update({str(i):1})
for i_file in range(len(finetune_tr_dl)):
    print("\nFile#", i_file + 1, "Finetuning")
    try:
        MTL.load_state_dict(torch.load("./MTL3_final_{}_{}.pth".format(date_list[-6],code_uq[i_file])))
        print("Load", "./MTL3_final_{}_{}.pth".format(date_list[-6],code_uq[i_file]))
        trEPOCH = 100
        teEPOCH = 200
    except Exception as e:
        print(str(e))
        MTL.load_state_dict(torch.load("./MTL3_base.pth"))
        print("Load Base")
        trEPOCH = 300
        teEPOCH = 400

    for p in MTL.parameters():
        p.requires_grad = True
    opft = [torch.optim.Adam(MTL.parameters(), lr=lrs[x]) for x in range(2)]
    epo_score = []
    no_up = 0
    max_va = -1
    for epo in range(trEPOCH):
        print("epoch#", epo + 1)
        scores = []
        for i_batch, sample_batched in enumerate(finetune_tr_dl[i_file]):
            for mini_batch in range(sample_batched['X'].size()[0]):
                tmp = []
                x = var(sample_batched['X'][mini_batch]).cuda()
                val = var(sample_batched['close'][mini_batch]).cuda()
                for i_sm in [2, 3, 4, 0, 1]:
                    cuda.empty_cache()
                    for p in MTL.parameters():
                        p.requires_grad = True

                    code_list, ans_list, updn = MTL(x, val)

                    opft[opctrl[str(i_sm)]].zero_grad()

                    y2 = var(torch.FloatTensor([[sample_batched['Y'][mini_batch][i_sm + 6]]])).cuda()
                    y3 = var(torch.FloatTensor(np.sign(np.array([sample_batched['Y'][mini_batch][5 + 1 + i_sm] - sample_batched['Y'][mini_batch][5 + 0 + i_sm]])))).view(1,1,-1).cuda()
                    score = (50 * ((y2 - torch.abs(ans_list[i_sm] - y2)) / y2) + 50 * lf4(updn[i_sm], y3))
                    loss = lf3((50 * ((y2 - torch.abs(ans_list[i_sm] - y2)) / y2) + 50 * lf4(updn[i_sm], y3)).view(1, -1), _100.view(1, -1))
                    loss.backward()

                    if i_sm not in [4, 3, 2]: # freeze all other parameters
                        for p in MTL.parameters():
                            p.requires_grad = False
                    else: # freeze all predictors and filters, but unlock CNN parts
                        for p in MTL.parameters():
                            p.requires_grad = True
                        for m in MTL.filter:
                            for p in m.parameters():
                                p.requires_grad = False
                        for m in MTL.predictor:
                            for p in m.module.parameters():
                                p.requires_grad = False
                    for p in MTL.filter[i_sm].parameters(): # unlock the filter we focused
                        p.requires_grad = True
                    for p in MTL.predictor[i_sm].module.parameters(): # unlock the predictor we focused
                        p.requires_grad = True

                    opft[opctrl[str(i_sm)]].step()
                    cuda.empty_cache()
                    tmp.append((score * w[i_sm]).view(-1).cpu().data.numpy()[0])
                scores.append(np.sum(tmp))
        epo_score.append(np.average(scores))
        print("score", epo_score[-1])
        valid_tmp = []
        for i_batch, sample_batched in enumerate(finetune_te_dl[i_file]):
            for mini_batch in range(sample_batched['X'].size()[0]):
                x = var(sample_batched['X'][mini_batch]).cuda()
                val = var(sample_batched['close'][mini_batch]).cuda()
                code_list, ans_list, updn = MTL(x, val)

                y2 = var(torch.FloatTensor([[sample_batched['Y'][mini_batch][0 + 6]]])).cuda()
                y3 = var(torch.FloatTensor(np.sign(np.array([sample_batched['Y'][mini_batch][5 + 1] - sample_batched['Y'][mini_batch][5 + 0]])))).view(1,1,-1).cuda()
                score = w[0] * (0.5 * ((y2 - torch.abs(ans_list[0] - y2)) / y2) + 0.5 * lf4(updn[0], y3))
                for i in range(1, len(code_list)):
                    y2 = var(torch.FloatTensor([[sample_batched['Y'][mini_batch][i + 6]]])).cuda()
                    y3 = var(torch.FloatTensor(np.sign(np.array([sample_batched['Y'][mini_batch][5 + 1 + i] - sample_batched['Y'][mini_batch][5 + 0 + i]])))).cuda()
                    score += w[i] * (0.5 * ((y2 - torch.abs(ans_list[i] - y2)) / y2) + 0.5 * lf4(updn[i], y3))

                valid_tmp.append(score.cpu().data.numpy()[0])
        print("Valid:", np.average(valid_tmp))
        if np.average(valid_tmp) > max_va:
            no_up = 0
            max_va = np.average(valid_tmp)
        else:
            no_up += 1
        if no_up >= 5:
            print("Fintune has done! Average Score is {0} / 1".format(np.average(epo_score) / 100))
            break
    pass
    print("Valid Model...")
    for p in MTL.parameters():
        p.requires_grad = False
    valid_tmp = []
    for i_batch, sample_batched in enumerate(finetune_te_dl[i_file]):
        for mini_batch in range(sample_batched['X'].size()[0]):
            x = var(sample_batched['X'][mini_batch]).cuda()
            val = var(sample_batched['close'][mini_batch]).cuda()
            code_list, ans_list, updn = MTL(x, val)

            y2 = var(torch.FloatTensor([[sample_batched['Y'][mini_batch][0 + 6]]])).cuda()
            y3 = var(torch.FloatTensor(np.sign(np.array([sample_batched['Y'][mini_batch][5 + 1] - sample_batched['Y'][mini_batch][5 + 0]])))).view(1,1,-1).cuda()
            score = w[0] * (0.5 * ((y2 - torch.abs(ans_list[0] - y2)) / y2) + 0.5 * lf4(updn[0], y3))
            for i in range(1, len(code_list)):
                y2 = var(torch.FloatTensor([[sample_batched['Y'][mini_batch][i + 6]]])).cuda()
                y3 = var(torch.FloatTensor(np.sign(np.array([sample_batched['Y'][mini_batch][5 + 1 + i] - sample_batched['Y'][mini_batch][5 + 0 + i]])))).cuda()
                score += w[i] * (0.5 * ((y2 - torch.abs(ans_list[i] - y2)) / y2) + 0.5 * lf4(updn[i], y3))

            valid_tmp.append(score.cpu().data.numpy()[0])
        print("Batch#{0}, Score = {1} / 1".format(i_batch, np.round(np.average(valid_tmp[-sample_batched['X'].size()[0] : ]), 6)))
    pass
    for p in MTL.parameters():
        p.requires_grad = True
    print("Fintune again for predict next week...")
    epo_score = []
    no_up = 0
    max_va = -1
    for epo in range(teEPOCH):
        print("epoch#", epo + 1)
        scores = []
        for i_batch, sample_batched in enumerate(finetune_te_dl[i]):
            for mini_batch in range(sample_batched['X'].size()[0]):
                tmp = []
                x = var(sample_batched['X'][mini_batch]).cuda()
                val = var(sample_batched['close'][mini_batch]).cuda()
                for i_sm in [2, 3, 4, 0, 1]:
                    cuda.empty_cache()
                    for p in MTL.parameters():
                        p.requires_grad = True

                    code_list, ans_list, updn = MTL(x, val)

                    opft[opctrl[str(i_sm)]].zero_grad()

                    y2 = var(torch.FloatTensor([[sample_batched['Y'][mini_batch][i_sm + 6]]])).cuda()
                    y3 = var(torch.FloatTensor(np.sign(np.array([sample_batched['Y'][mini_batch][5 + 1 + i_sm] - sample_batched['Y'][mini_batch][5 + 0 + i_sm]])))).view(1,1,-1).cuda()
                    score = (50 * ((y2 - torch.abs(ans_list[i_sm] - y2)) / y2) + 50 * lf4(updn[i_sm], y3))
                    loss = lf3((50 * ((y2 - torch.abs(ans_list[i_sm] - y2)) / y2) + 50 * lf4(updn[i_sm], y3)).view(1, -1), _100.view(1, -1))
                    loss.backward()

                    if i_sm not in [4, 3, 2]: # freeze all other parameters
                        for p in MTL.parameters():
                            p.requires_grad = False
                    else: # freeze all predictors and filters, but unlock CNN parts
                        for p in MTL.parameters():
                            p.requires_grad = True
                        for m in MTL.filter:
                            for p in m.parameters():
                                p.requires_grad = False
                        for m in MTL.predictor:
                            for p in m.module.parameters():
                                p.requires_grad = False
                    for p in MTL.filter[i_sm].parameters(): # unlock the filter we focused
                        p.requires_grad = True
                    for p in MTL.predictor[i_sm].module.parameters(): # unlock the predictor we focused
                        p.requires_grad = True

                    opft[opctrl[str(i_sm)]].step()
                    cuda.empty_cache()
                    tmp.append((score * w[i_sm]).cpu().data.numpy()[0])
                scores.append(np.sum(tmp))
        epo_score.append(np.average(scores))
        print("score", epo_score[-1])#93fcff
        if epo_score[-1] > max_va:
            no_up = 0
            max_va = epo_score[-1]
        else:
            no_up += 1
        if no_up >= 3:
            print("Fintune has done! Average Score is {0} / 1".format(np.average(epo_score) / 100))
            break
    pass
    print("Prepare to predict")
    raw = pd.read_csv("./DataSet/final18_"+str(date_list[-1])+".csv")
    val = raw.values
    thisweek = []
    idx = 0
    for i in range(1, len(val)):
        if val[i][1] != val[idx][1]:
            thisweek.append(val[i - 9:i])
            idx = i
    thisweek.append(val[len(val) - 9:])
    thisweek2 = []
    for i in range(len(thisweek)):
        tmp = []
        for ii in range(len(thisweek[i])):
            tmp.append(thisweek[i][ii][13:])
        thisweek2.append(tmp)

    x = var(torch.FloatTensor(thisweek2[i_file])).view(1, 9, -1).cuda()
    val = var(torch.FloatTensor([thisweek[i_file][-1][7]])).cuda()
    code_list, ans_list, updn =MTL(x, val)

    res = []
    res.append(code_uq[i_file])
    for ans, ud in zip(ans_list, updn):
        res.append(ud.view(-1).cpu().data.numpy()[0])
        res.append(ans.view(-1).cpu().data.numpy()[0])
    Result.append(res)
    torch.save(MTL.state_dict(), "./MTL3_final_{0}_{1}.pth".format(date_list[-1], code_uq[i_file]))
    cuda.empty_cache()
    print("Result:", res)

result = pd.DataFrame(Result, columns=["ETFid","Mon_ud","Mon_cprice","Tue_ud","Tue_cprice", "Wed_ud","Wed_cprice","Thu_ud","Thu_cprice","Fri_ud","Fri_cprice"])
result.to_csv("./result_"+str(date_list[-1])+".csv", index=False)
