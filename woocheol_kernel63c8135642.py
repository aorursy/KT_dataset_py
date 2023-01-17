# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
 
# Any results you write to the current directory are saved as output.
sub = pd.read_csv('../input/submission2/submission.csv')
sub.to_csv('practice1.csv',index=False)
sub = pd.read_csv('../input/pleasetry1/sub1.csv')
sub.to_csv('sub1.csv',index=False)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
sub = pd.read_csv('../input/pleasetry1/sub1.csv')
sub.to_csv('sub1.csv',index=False)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

sub = pd.read_csv('../input/pleasetry2/sub2.csv')
sub.to_csv('sub2.csv',index=False)
import numpy as np # linear algebra
import pandas as pd
sub = pd.read_csv('../input/pleasetry3/sub3.csv')
sub.to_csv('sub3.csv',index=False)
import numpy as np # linear algebra
import pandas as pd
sub = pd.read_csv('../input/pleasetry4/sub3.csv')
sub.to_csv('sub4.csv',index=False)
import numpy as np # linear algebra
import pandas as pd # data processing
sub = pd.read_csv('../input/pleasetry5/sub5.csv')
sub.to_csv('sub5.csv',index=False)
import numpy as np # linear algebra
import pandas as pd
sub = pd.read_csv('../input/pleasetry6/sub5.csv')
sub.to_csv('sub5.csv',index=False)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
sub = pd.read_csv('../input/lastTry1/sub6.csv')
sub.to_csv('sub6.csv',index=False)
!ls ../input/
!ls ../input/
sub = pd.read_csv('../input/lasttry1/sub6.csv')
sub.to_csv('sub6.csv',index=False)
sub = pd.read_csv('../input/lasttry1/sub6_avg.csv')
sub.to_csv('sub6_avg.csv',index=False)
