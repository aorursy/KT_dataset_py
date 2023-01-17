import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
!wget -q -O other1.csv https://www.kaggleusercontent.com/kf/36961390/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..k0-Yl9R-H0lC86Nth0mIMQ.HiygAXAzwROTh0HZTHHGexh-f2l5eaPH1wFhHwsZxcrhqsddvY1Vrvy8xnKkzleMoYKR2DFzwKr3wpX3JkzC74BsxlPSpA5WeSAJkUitCktmGkcdB-EOiusI6rEBYMJUaXgx52hzF-dcVTPgRlj2HgnLCixBfs78h7mgFokLo1kZjgXjWlDHXcRao9DGKKHvp2EZxkGJ3-crDI_7kLxFfpDDlUjxmtLSZTk6yNxnExsj-nljUOan0tu7dHihhrrABxR7ZeT55MKqhkqew8slishHNIfnUvZug-n50E_yf0fjh8ydo7LUVYOVT4oY3WllNNhDc1PIEWXau9MPJITQZ6zW_KxYt9qRPkfCrXAr8UXtffECQ7Ut0zUcn7CUo1qen4fBLVhHnixAPnD6rFlv_23qvSgWRXbHQSVcjlSpC2tKWfVTFswN2zRREiahEiVbxWRH3vDrHk0uJeWSmHPUqS7cYT71h0bZhtmuOWd8T5hhk1hC4tECcSuZvxG80woxHsitF0YiQjDFQTTL7T3vUBSQYzLeAnTs4N5iZNVpyaIBVJJDCQ76yZ1lvDK9126jMASTxwnZVjb5mUAuYTpkxpS10tg-V8pzIsyoqitRAgWYiiMZ-AWdofBltCqZbW6ZCAshq9jSwr9dhjoDt1H5PD6m8A4Sko8i0oGH4YqoGMk.XzhuO7hsNjxzjnivzC2boA/sub_EfficientNetB3_384.csv
a = pd.read_csv('other1.csv')
b = pd.read_csv('../input/stack-models/submission.csv')
c = pd.read_csv('../input/melanoma-efficientnet-b6-tpu-tta/submission.csv')
pd.DataFrame({'image_name': a.image_name, 'target': (a.target + c.target*0.8)/2}).to_csv('submission.csv', index=False)
