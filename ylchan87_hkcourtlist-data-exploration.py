# The sqlAlchemy datamodel from

# https://github.com/ylchan87/HKCourtList/blob/master/dataModel.py



import sqlalchemy



from sqlalchemy import create_engine

from sqlalchemy.orm import sessionmaker

from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()



from sqlalchemy import Column, Integer, String, DateTime

from sqlalchemy import ForeignKey

from sqlalchemy.orm import relationship



from sqlalchemy import Table



global session

session = None

def init(sqlPath='sqlite:///:memory:', echo=False):

    engine = create_engine(sqlPath, echo=echo)

    Base.metadata.create_all(engine)

    Session = sessionmaker(bind=engine)

    global session

    session = Session()

    return session



def get_or_create(cls, **kwargs):

    if not session: 

        print('db not init/connected yet')

        return None



    instance = session.query(cls).filter_by(**kwargs).first()

    if not instance:

        instance = cls(**kwargs)

        session.add(instance)

        session.flush([instance])

    return instance



# association table for many to many relationships

events_judges = Table('events_judges', Base.metadata,

    Column('event_id', ForeignKey('events.id'), primary_key=True),

    Column('judge_id', ForeignKey('judges.id'), primary_key=True)

)



events_cases = Table('events_cases', Base.metadata,

    Column('event_id', ForeignKey('events.id'), primary_key=True),

    Column('case_id', ForeignKey('cases.id'), primary_key=True)

)



events_tags = Table('events_tags', Base.metadata,

    Column('event_id', ForeignKey('events.id'), primary_key=True),

    Column('tag_id', ForeignKey('tags.id'), primary_key=True)

)



events_lawyers = Table('events_lawyers', Base.metadata,

    Column('event_id', ForeignKey('events.id'), primary_key=True),

    Column('lawyer_id', ForeignKey('lawyers.id'), primary_key=True)

)



events_lawyers_atk = Table('events_lawyers_atk', Base.metadata,

    Column('event_id', ForeignKey('events.id'), primary_key=True),

    Column('lawyer_id', ForeignKey('lawyers.id'), primary_key=True)

)



events_lawyers_def = Table('events_lawyers_def', Base.metadata,

    Column('event_id', ForeignKey('events.id'), primary_key=True),

    Column('lawyer_id', ForeignKey('lawyers.id'), primary_key=True)

)



class Event(Base):

    __tablename__ = 'events'

    id = Column(Integer, primary_key=True)

    category = Column(String)

    court = Column(String)



    judges = relationship("Judge", 

                          secondary=events_judges,

                          back_populates='events')



    datetime = Column(DateTime(), nullable=True)



    #sometimes a event can have 2 cases

    cases = relationship("Case", 

                          secondary=events_cases,

                          back_populates='events')



    parties = Column(String)

    parties_atk = Column(String)

    parties_def = Column(String)



    tags = relationship("Tag", 

                          secondary=events_tags,

                          back_populates='events')



    lawyers = relationship("Lawyer", 

                          secondary=events_lawyers,

                          back_populates='events')



    lawyers_atk = relationship("Lawyer", 

                          secondary=events_lawyers_atk,

                          back_populates='events_atk')

    

    lawyers_def = relationship("Lawyer", 

                          secondary=events_lawyers_def,

                          back_populates='events_def')



    @classmethod

    def get_or_create(cls, **kwargs):

        return get_or_create(cls, **kwargs)



    def __repr__(self):

        return "<Event(category='%s', datetime='%s')>" % (

                            self.category, self.datetime)

    

    def fullDesc(self):

        print("category   :", self.category   )

        print("court      :", self.court      )

        print("judges     :", self.judges     )

        print("datetime   :", self.datetime   )

        print("cases      :", self.cases      )

        print("parties    :", self.parties    )

        print("parties_atk:", self.parties_atk)

        print("parties_def:", self.parties_def)

        print("tags       :", self.tags       )

        print("lawyers    :", self.lawyers    )

        print("lawyers_atk:", self.lawyers_atk)

        print("lawyers_def:", self.lawyers_def)



class Judge(Base):

    __tablename__ = 'judges'

    id = Column(Integer, primary_key=True)

    name_zh = Column(String, unique=False)

    name_en = Column(String, unique=False)



    events = relationship("Event", 

                          secondary=events_judges,

                          back_populates='judges')



    @classmethod

    def get_or_create(cls, **kwargs):

        return get_or_create(cls, **kwargs)



    def __repr__(self):

        return "<Judge(name_zh='%s', name_en='%s')>" % (

                            self.name_zh, self.name_en)



class Case(Base):

    __tablename__ = 'cases'

    id = Column(Integer, primary_key=True)

    caseNo = Column(String, unique=True)

    description = Column(String)

    events = relationship("Event", 

                          secondary=events_cases,

                          back_populates='cases')



    @classmethod

    def get_or_create(cls, **kwargs):

        return get_or_create(cls, **kwargs)

    

    def __repr__(self):

        return "<Case(caseNo='%s', description='%s')>" % (

                            self.caseNo, self.description)



class Tag(Base):

    """

    This correspond to 'Offence' 'Offence/Nature' and 'Hearing' column

    """

    __tablename__ = 'tags'

    id = Column(Integer, primary_key=True)

    name_zh = Column(String, unique=False)

    name_en = Column(String, unique=False)



    events = relationship("Event", 

                          secondary=events_tags,

                          back_populates='tags')



    @classmethod

    def get_or_create(cls, **kwargs):

        return get_or_create(cls, **kwargs)



    def __repr__(self):

        return "<Tag(name_zh='%s', name_en='%s')>" % (

                            self.name_zh, self.name_en)



class Lawyer(Base):

    __tablename__ = 'lawyers'

    id = Column(Integer, primary_key=True)

    name_zh = Column(String, unique=False) # '孖士打律師行' = 'Mayer Brown' also 'Mayer Brown JSM'...

    name_en = Column(String, unique=False)

    

    events = relationship("Event", 

                          secondary=events_lawyers,

                          back_populates='lawyers')



    events_atk = relationship("Event", 

                          secondary=events_lawyers_atk,

                          back_populates='lawyers_atk')

    

    events_def = relationship("Event", 

                          secondary=events_lawyers_def,

                          back_populates='lawyers_def')



    @classmethod

    def get_or_create(cls, **kwargs):

        return get_or_create(cls, **kwargs)

    

    def __repr__(self):

        return "<Lawyer(name_zh='%s', name_en='%s')>" % (

        self.name_zh, self.name_en)
# download file with "requests"

# https://stackoverflow.com/questions/16694907/download-large-file-in-python-with-requests

import requests

import shutil



def download_file(url, local_filename):

    r = requests.get(url, stream=True)

    with open(local_filename, 'wb') as f:

        shutil.copyfileobj(r.raw, f)
# plotly standard imports

import plotly.graph_objs as go

import plotly.plotly as py



# Cufflinks wrapper on plotly

import cufflinks



# Data science imports

import pandas as pd

import numpy as np



from plotly.offline import iplot

cufflinks.go_offline()
download_file("https://morph.io/ylchan87/HKCourtList/data.sqlite?key=tujNIuYznWUaPWxO8ltL", "data.sqlite")
session = init("sqlite:///data.sqlite") #init sqlAlchemy datamodel
events = session.query(Event).all()

events[0].fullDesc()
cs = session.query(Case).filter_by(description="婚姻訴訟").all()

lawyers_count = {}

for c in cs:

    case_lawyers = set()

    for e in c.events:

        for l in e.lawyers + e.lawyers_atk + e.lawyers_def: case_lawyers.add(l)

    for l in case_lawyers:

        try:

            lawyers_count[l]+=1

        except KeyError:

            lawyers_count[l]=1
df = (pd.DataFrame([ [l.name_zh, l.name_en, c] for l,c in lawyers_count.items()], columns = ["name_zh", "name_en", "count"])

      .sort_values("count", ascending=False)

     )

df.head(10)
df.iplot(kind='pie', labels="name_en", values="count", title="Lawyers employed in case of type 婚姻訴訟")
lawyers = session.query(Lawyer).all()

df = (pd.DataFrame([ [l.name_zh, l.name_en, len(l.events)] for l in lawyers], columns=["name_zh","name_en","nEvents"])

      .sort_values("nEvents",ascending=False)

     )

df.head(10)
aLawyer = session.query(Lawyer).filter_by( name_en="Yip, Tse & Tang").first()
df = pd.DataFrame([e.category for e in aLawyer.events])

cat_counts = df[0].value_counts()

cat_counts = pd.DataFrame(cat_counts).reset_index()

cat_counts.columns = ['courtType','count']

cat_counts.iplot(kind='pie', labels='courtType', values='count', title="Court type of cases by lawyer '%s'" % aLawyer.name_en)