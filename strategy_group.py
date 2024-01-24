import pickle
import numpy as np
import pandas as pd

def drawdown(cum):
    cumulative = cum+1
    highwatermark = cumulative.cummax()
    drawdown = (cumulative / highwatermark) -1
    return drawdown
def max_dd(cum):
    return np.min(drawdown(cum))

def cagr(cash_hist):
    return (cash_hist[-1]/cash_hist[0])**(252/len(cash_hist))-1

def stdv(returns):
    return np.std(returns)*math.sqrt(252)

def sharpe(returns):
    return cagr(returns.cumprod())/stdv(returns)

def winsorize(x):
    if x==0:
        return 1
    elif (x>0.69) and (x<1.31):
        return x
    else:
        return 1
        
def get_item(data_id):
    if data_id == 'close':
        return close_df
    elif data_id == 'open':
        return open_df
    elif data_id == 'high':
        return high_df
    elif data_id == 'low':
        return low_df
    elif data_id == 'volume':   #거래량
        return acml_df
    elif data_id == 'return':
        return return_df
    else:                          ## 거래대금
        return volume_df

with open('./close_df.pkl', 'rb') as f:
    close_df = pickle.load(f)
with open('./volume_df.pkl', 'rb') as f:
    volume_df = pickle.load(f)
with open('./acml_df.pkl', 'rb') as f:
    acml_df = pickle.load(f)
with open('./low_df.pkl', 'rb') as f:
    low_df = pickle.load(f)
with open('./high_df.pkl', 'rb') as f:
    high_df = pickle.load(f)
with open('./open_df.pkl', 'rb') as f:
    open_df = pickle.load(f)
with open('./return_df.pkl', 'rb') as f:
    return_df = pickle.load(f)
    return_df = (return_df-return_df+1)*((1+return_df).applymap(winsorize)-1)

close_df = close_df.loc['2005-01-01':,:]
volume_df = volume_df.loc['2005-01-01':,:]
acml__df = acml_df.loc['2005-01-01':,:]
low_df = low_df.loc['2005-01-01':,:]
high_df = high_df.loc['2005-01-01':,:]
open_df = open_df.loc['2005-01-01':,:]
return_df = return_df.loc['2005-01-01':,:]

class st_group(object):
    def __init__(self):
        self.returns = get_item('return')
        self.close = get_item('close')
        self.open = get_item('open')
        self.high = get_item('high')
        self.low = get_item('low')
        self.trade_vol = get_item('volume') ##거래량
        self.trade_amount = get_item('amount of money') ##거래대금
        self.vwap = self.trade_amount / self.trade_vol
        self.adv20 = self.trade_amount.rolling(20).mean()
        self.isEW = True
        self.isascending=True
        self.isupperratio=0.5
        self.weight=None
        self.delta_weight=None
        self.performance = None
        
    def cal_weight(self, formula, isEW=True, isascending=True, isupperratio=0):
        #isEW==True : eqeually weighted, False : Value weighted
        #isascending==False : when isEW==False, formula 결과 낮은 값의 종목에 더 많은 비중  //  isascending==True : formula 결과 높은 값의 종목에 더 많은 비중
        ### 참고: rank(ascending==True)함수는 가장 작은 값을 1등으로 만듦
        #isupperratio==float btw [0,1] : decision of ratio for target levels. The lower, the broader universe
        
        #Set universe (mask)
        ratio = formula.apply(lambda x: x.rank(ascending=isascending)/x.rank(ascending=isascending).max(),axis=1)
        formula2 = (ratio>isupperratio)*1 
        
        #Calculate weight
        if isEW==True:
            weight = formula2.apply(lambda x: x/x.sum(),axis=1)
        elif isEW==False:
            formula3 = formula2.replace(0, np.NaN) #to avoid zero becomes a meaningful expression
            #formula값이 음수~양수로 있을 때, formula값이 0인 건 rank에 영향을 줌. 근데 mask인 formula2를 통해서 음수or양수가 0이라는 rank에 영향을 주는 의미있는 수를 갖는 건 안 되기 때문에 nan처리함
            formula4 = formula * formula3 
            weight = formula4.apply(lambda x: x.rank(ascending=isascending)/x.rank(ascending=isascending).sum(), axis=1)
        weight = weight.replace(np.NaN, 0) #to calculate accurrate delta_weight
        weight_update = ((weight * (1+self.returns)).T/(weight * (1+self.returns)).sum(1)).T
        self.weight = weight_update.shift(1)
        self.delta_weight = self.weight - weight_update
        self.performance = ((self.weight.loc['2006-01-01':'2019-01-01',:] * (self.returns.loc['2006-01-01':'2019-01-01',:]+1)).sum(axis=1)-np.abs((self.delta_weight.loc['2006-01-01':'2019-01-01',:]<0)*self.delta_weight.loc['2006-01-01':'2019-01-01',:]).sum(1)*0.0031)  

        print(self.isEW, self.isascending, self.isupperratio)
        print(self.performance.describe())
        print('Final cumulative returns is %s' % self.performance.cumprod().values[-1])
        print('cagr: {} ,  sharpe: {} ,  mdd: {} ,  Total transaction cost: {}'.format(
            cagr(self.performance.cumprod(), 13), sharpe(self.performance-1, 13, 252),
            max_dd(self.performance.cumprod()), (np.abs((self.delta_weight.loc['2006-01-01':'2019-01-01',:]<0)*self.delta_weight.loc['2006-01-01':'2019-01-01',:]).sum(1)*0.0031).sum()
        ))
        
    def st0(self):
        formula = (self.close-self.low) / (self.high - self.low)
        self.cal_weight(formula, self.isEW, self.isascending, self.isupperratio)
        
    def st1(self):
        temp = (self.returns<0)*self.returns.rolling(20).std()**2 + (self.returns>=0)*self.close**2     
        temp2 = temp.rolling(5).max()
        formula = temp2.rank(1) - 0.5
        self.cal_weight(formula, self.isEW, self.isascending, self.isupperratio)
    

    def st2(self):
        temp = (np.log(self.trade_vol).replace([np.inf, -np.inf], np.nan) - np.log(self.trade_vol.shift(2))).rank(1)
        temp2 = ((self.close-self.open)/self.open).rank(1) 
        formula = -temp.rolling(6).corr(temp2)
        self.cal_weight(formula, self.isEW, self.isascending, self.isupperratio)
        
        
    def st3(self):
        temp = self.open.rank(1)
        temp2 = self.trade_vol.rank(1)
        formula = -temp.rolling(10).corr(temp2)
        self.cal_weight(formula, self.isEW, self.isascending, self.isupperratio)
        
    def st4(self):
        formula = -(self.low).rolling(9).rank()
        self.cal_weight(formula, self.isEW, self.isascending, self.isupperratio)
    
    def st5(self):
        # FT5: cagr: 0.06258305465556657 ,  sharpe: 0.3245969225143553 ,  mdd: -0.37210144272
        # TT5: cagr: -0.006394098898749978 ,  sharpe: -0.03128703268207805 ,  mdd: -0.3828
        formula = (self.open - self.vwap.rolling(10).sum()/10).rank(1)*(-1* np.abs(self.close-self.vwap).rank(1))
        self.cal_weight(formula, self.isEW, self.isascending, self.isupperratio)
        
    def st6(self):
        formula = -self.open.rolling(10).corr(self.trade_vol)
        self.cal_weight(formula, self.isEW, self.isascending, self.isupperratio)

    def st7(self):
        delta_7_close = self.close-self.close.shift(7)
        formula = (self.trade_amount.rolling(20).mean()<self.trade_amount)* np.abs(delta_7_close).rolling(60).rank()*(-np.sign(delta_7_close)) + ((self.trade_amount.rolling(20).mean()>=self.trade_amount))*(-1)     
        self.cal_weight(formula, self.isEW, self.isascending, self.isupperratio)

    
    def st8(self):  ##mean-reversion 
        temp=(self.open.rolling(5).sum() * self.returns.rolling(5).sum())
        formula = -(temp-temp.shift(10)).rank(1)  #10일 전보다 temp값이 높으면 1등 부여. 즉, 작을수록 큰 비중
        self.cal_weight(formula, self.isEW, self.isascending, self.isupperratio)

    def st_marketEW(self):  ##어떤 하이퍼파라미터를 넣든 얘는 전체 EW로 나옴
        # cagr 0.105134  sharpe 0.55240  mdd -0.3564
        formula = self.open.notna().astype(int).replace(0, np.NaN)
        self.cal_weight(formula, self.isEW, self.isascending, self.isupperratio)
    
    def st_s1(self): ##하이퍼파라미터에 관게없이 해당 조건을 만족하는 EW로 나옴
        formula = (self.low > self.high.shift(1))*1
        self.cal_weight(formula, self.isEW, self.isascending, self.isupperratio)
        
    def st_s2(self): ##하이퍼파라미터에 관게없이 얘는 해당 조건을 만족하는 EW로 나옴
        formula = (self.trade_vol > self.trade_vol.shift(1)*2)*1
        self.cal_weight(formula, self.isEW, self.isascending, self.isupperratio)
        
    def st_s3(self): ##하이퍼파라미터에 관게없이 얘는 해당 조건을 만족하는 EW로 나옴
        formula = (self.low - self.close.pct_change(periods=2)*self.close/self.close > self.close)*1
        self.cal_weight(formula, self.isEW, self.isascending, self.isupperratio)

    def st9(self):
        delta_close_1 = self.close-self.close.shift(1)
        formula = (delta_close_1.rolling(5).min()<0)*delta_close_1+(delta_close_1.rolling(5).min()>=0)*(self.close.shift(1).rolling(5).max()<0)*delta_close_1+(self.close.shift(1).rolling(5).min()>=0)*(self.close.shift(1).rolling(5).max()>=0)*(-1*delta_close_1)
        self.cal_weight(formula, self.isEW, self.isascending, self.isupperratio)
        
    def st10(self):
        delta_close_1 = self.close-self.close.shift(1)
        formula = (delta_close_1.rolling(4).min()>0)*delta_close_1+(delta_close_1.rolling(4).min()<=0)* (delta_close_1.rolling(4).max()<0)*delta_close_1+(delta_close_1.rolling(4).min()<=0)* (delta_close_1.rolling(4).max()>=0)*(-1)*(delta_close_1)
        self.cal_weight(formula, self.isEW, self.isascending, self.isupperratio)
        
    def st11(self):
        formula = (self.vwap-self.close).rolling(3).max().rank(1)+(self.vwap-self.close).rolling(3).min().rank(1)*(self.trade_vol - self.trade_vol.shift(3)).rank(1)
        self.cal_weight(formula, self.isEW, self.isascending, self.isupperratio)
    
    def st12(self):
        delta_vol_1 = self.trade_vol - self.trade_vol.shift(1)
        delta_close_1 = self.close - self.close.shift(1)
        formula = (delta_vol_1<0)*delta_close_1+(delta_vol_1>0)*(-1)*delta_close_1
        self.cal_weight(formula, self.isEW, self.isascending, self.isupperratio)
        
    def st13(self):
        formula = (-1)*(self.close.rolling(5).cov(self.trade_vol)).rank(1)
        self.cal_weight(formula, self.isEW, self.isascending, self.isupperratio)
    
    def st14(self):
        formula = (-1)*(self.returns-self.returns.shift(3)).rank(1)*(self.open.rolling(10).corr(self.trade_vol))
        self.cal_weight(formula, self.isEW, self.isascending, self.isupperratio)
        
    def st15(self):
        temp = (self.high.rolling(3).corr(self.trade_vol)).rank(1)
        formula = (-1)*temp.rolling(3).sum().rank(1)
        self.cal_weight(formula, self.isEW, self.isascending, self.isupperratio)
        
    def st16(self):
        formula = (-1)*(self.high.rolling(5).cov(self.trade_vol)).rank(1)
        self.cal_weight(formula, self.isEW, self.isascending, self.isupperratio)
    
    def st17(self):
        delta_close_1 = self.close-self.close.shift(1)
        formula = (-1)*(self.close.rolling(10).rank()).rank(1)*(delta_close_1-delta_close_1.shift(1).rank(1))*((self.trade_vol/self.adv20).rolling(5).rank()).rank(1)
        self.cal_weight(formula, self.isEW, self.isascending, self.isupperratio)

    def st18(self):
        formula = (-1)*np.abs(self.close-self.open).rolling(5).std().rank(1) + (self.close-self.open)+self.close.rolling(10).corr(self.open)
        self.cal_weight(formula, self.isEW, self.isascending, self.isupperratio)

    def st19(self):
        delta_close_7 = self.close - self.close.shift(7)
        formula = (delta_close_7>0)*(-1)*(1+(1+self.returns.rolling(250).sum()).rank(1))+(delta_close_7<0)*(1+(1+self.returns.rolling(250).sum()).rank(1))
        self.cal_weight(formula, self.isEW, self.isascending, self.isupperratio)
    
    def st20(self):
        formula = (-1)*(self.open - self.high.shift(1)).rank(1)*(self.open-self.close.shift(1)).rank(1)*(self.open-self.low.shift(1)).rank(1)
        self.cal_weight(formula, self.isEW, self.isascending, self.isupperratio)
    
    def st21(self):
        close_8_mean = self.close.rolling(8).mean()
        close_8_std = self.close.rolling(8).std()
        close_2_mean = self.close.rolling(2).mean()
        cond1 = close_8_mean + close_8_std < close_2_mean
        cond2 = close_2_mean < (close_8_mean - close_8_std)
        cond3 = 1<=self.trade_amount/self.adv20
        formula = (cond1==True)*0 + (cond1==False)*(cond2==True)*1 + (cond1==False)*(cond2==False)*(cond3==True)*1+ (cond1==False)*(cond2==False)*(cond3==False)*0
        self.cal_weight(formula, self.isEW, self.isascending, self.isupperratio)

    def st22(self):
        hv_corr_5 = self.high.rolling(5).corr(self.trade_vol)
        formula = (-1)*(hv_corr_5-hv_corr_5.shift(5))*(self.close.rolling(20).std().rank(1))
        self.cal_weight(formula, self.isEW, self.isascending, self.isupperratio)
    
    def st23(self):
        cond1= self.high.rolling(20).mean() < self.high
        formula = cond1*(-1)*(self.high-self.high.shift(2))
        self.cal_weight(formula, self.isEW, self.isascending, self.isupperratio)
        
    def st24(self):
        close_100_mean = self.close.rolling(100).mean()
        cond1 = (close_100_mean - close_100_mean.shift(100))/close_100_mean.shift(100) <=0.05
        formula = cond1*(-1)*(self.close-self.close.rolling(100).min())+(1-cond1)*(-1)*(self.close-self.close.shift(3))
        self.cal_weight(formula, self.isEW, self.isascending, self.isupperratio)

    def st25(self):
        formula = (-1)*self.returns * self.adv20 * self.vwap * (self.high-self.close)
        self.cal_weight(formula, self.isEW, self.isascending, self.isupperratio)

    def st26(self):
        vol_5_rank = self.trade_vol.rolling(5).rank()
        high_5_rank = self.high.rolling(5).rank()
        formula = (-1)*(vol_5_rank.rolling(5).corr(high_5_rank)).rolling(3).max()
        self.cal_weight(formula, self.isEW, self.isascending, self.isupperratio)

    def st27(self):
        temp = self.trade_vol.rolling(6).corr(self.vwap)
        temp2 = temp.rolling(2).mean()
        cond1= temp2.apply(lambda x: x.rank()/x.rank().max(), axis=1)>0.5
        formula = (cond1==True)*0+(cond1==False)*1
        self.cal_weight(formula, self.isEW, self.isascending, self.isupperratio)
        
    def st28(self):
        temp = self.adv20.rolling(5).corr(self.low)+(self.high+self.low)/2 - self.close
        formula = temp.apply(lambda x: x/np.abs(x).sum(), axis=1)
        self.cal_weight(formula, self.isEW, self.isascending, self.isupperratio)
        
    def st29(self):
        temp = (self.close-self.close.shift(5)).rank(1, ascending=False)
        temp2 = temp.rolling(2).min()
        temp3 = np.log(temp2).apply(lambda x: x/np.abs(x).sum(), axis=1)
        formula = (temp3.rank(1)).rolling(5).min() + ((-self.returns).shift(6)).rolling(5).rank()
        self.cal_weight(formula, self.isEW, self.isascending, self.isupperratio)

    def st30(self):
        delta_close_1 = self.close - self.close.shift(1)
        temp = np.sign(delta_close_1)+np.sign(delta_close_1.shift(1))+np.sign(delta_close_1.shift(2))
        formula = (1-temp.rank(1))*(self.trade_vol.rolling(5).sum()/self.trade_vol.rolling(20).sum())
        self.cal_weight(formula, self.isEW, self.isascending, self.isupperratio)
        
    def st32(self):
        temp = self.close.rolling(7).mean()-self.close
        temp2 = temp.apply(lambda x: x/np.abs(x).sum())
        temp3 = self.vwap.rolling(230).corr(self.close.shift(5))
        formula = temp2 + temp3.apply(lambda x: 20*x/np.abs(x).sum())
        self.cal_weight(formula, self.isEW, self.isascending, self.isupperratio)

    def st33(self):
        formula = (self.open/self.close -1).rank(1)
        self.cal_weight(formula, self.isEW, self.isascending, self.isupperratio)

    def st34(self):
        formula = (1-(self.returns.rolling(2).std()/self.returns.rolling(5).std()).rank(1)).rank(1)+(1-((self.close-self.close.shift(1)).rank(1)))
        self.cal_weight(formula, self.isEW, self.isascending, self.isupperratio)
