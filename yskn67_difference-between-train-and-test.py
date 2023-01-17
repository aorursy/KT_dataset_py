import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
dtype = {
    "ip": "uint32",
    "app": "uint16",
    "device": "uint16",
    "os": "uint16",
    "channel": "uint16",
}
train = pd.read_csv("../input/train.csv", dtype=dtype)
test = pd.read_csv("../input/test.csv", dtype=dtype)
fig, (axl, axr) = plt.subplots(ncols=2, figsize=(15,6))
axl.set_title("train")
axr.set_title("test")
sns.distplot(train["ip"], kde=False, ax=axl)
sns.distplot(test["ip"], kde=False, ax=axr)
plt.plot()
df = pd.concat([train["ip"].describe(), test["ip"].describe()], axis=1)
df.columns = ["train", "test"]
df
# train only
len(set(train["ip"]) - set(test["ip"]))
# test only
len(set(test["ip"]) - set(train["ip"]))
fig, (axl, axr) = plt.subplots(ncols=2, figsize=(15,6))
axl.set_title("train")
axr.set_title("test")
sns.distplot(train["app"], kde=False, ax=axl)
sns.distplot(test["app"], kde=False, ax=axr)
plt.plot()
df = pd.concat([train["app"].describe(), test["app"].describe()], axis=1)
df.columns = ["train", "test"]
df
# train only
len(set(train["app"]) - set(test["app"]))
# test only
len(set(test["app"]) - set(train["app"]))
fig, (axl, axr) = plt.subplots(ncols=2, figsize=(15,6))
axl.set_title("train")
axr.set_title("test")
sns.distplot(train["device"], kde=False, ax=axl)
sns.distplot(test["device"], kde=False, ax=axr)
plt.plot()
df = pd.concat([train["device"].describe(), test["device"].describe()], axis=1)
df.columns = ["train", "test"]
df
# train only
len(set(train["device"]) - set(test["device"]))
# test only
len(set(test["device"]) - set(train["device"]))
fig, (axl, axr) = plt.subplots(ncols=2, figsize=(15,6))
axl.set_title("train")
axr.set_title("test")
sns.distplot(train["os"], kde=False, ax=axl)
sns.distplot(test["os"], kde=False, ax=axr)
plt.plot()
df = pd.concat([train["os"].describe(), test["os"].describe()], axis=1)
df.columns = ["train", "test"]
df
# train only
len(set(train["os"]) - set(test["os"]))
# test only
len(set(test["os"]) - set(train["os"]))
fig, (axl, axr) = plt.subplots(ncols=2, figsize=(15,6))
axl.set_title("train")
axr.set_title("test")
sns.distplot(train["channel"], kde=False, ax=axl)
sns.distplot(test["channel"], kde=False, ax=axr)
plt.plot()
df = pd.concat([train["channel"].describe(), test["channel"].describe()], axis=1)
df.columns = ["train", "test"]
df
# train only
len(set(train["channel"]) - set(test["channel"]))
# test only
len(set(test["channel"]) - set(train["channel"]))
# train
pd.to_datetime(train["click_time"]).describe()
# test
pd.to_datetime(test["click_time"]).describe()