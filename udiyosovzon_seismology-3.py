! pip install obspy
import obspy as obs
# read all 3 traces into a Stream object

st = obs.read("../input/JRC2*sac")



# low pass filter

st = st.filter("lowpass", freq=0.1)



# make a copy of st called st_r. st_r will be rotated, st remains the same

st_r = st.copy()



# rotate horizontal components (N)orth and (E)ast to (R)adial and (T)ransverse

st_r.rotate("NE->RT", 150)



# take earth quake time and create a osbpy UTC object

eq_time = obs.UTCDateTime("2010-02-27 06:34:11")



# plot seismograms with time axis in seconds relative to eq_time

_ = st_r.plot(starttime=eq_time, type="relative")
# high pass filter

st = st.filter("highpass", freq=2)



# plot

_ = st.plot(starttime=eq_time, type="relative")
# slice seismogram to focus only on the dominant event

local_event_start = eq_time + 2180

local_event_end = eq_time + 2240

major_event = st.slice(starttime=local_event_start, endtime=local_event_end)



# plot

fig = major_event.plot(starttime=local_event_start, type="relative")
_ = major_event[2].plot(starttime=local_event_start, endtime=local_event_start+15, type="relative")
_ = major_event[:2].plot(starttime=local_event_start, endtime=local_event_start+20, type="relative")
import plotly
plotly.tools.mpl_to_plotly(fig)
# determine love wave start time

love_wave_time = eq_time + 2180



# slice seismogram to present only love wave

love_wave = st_r.slice(starttime=love_wave_time, endtime=love_wave_time+120)
_ = love_wave[0].plot(starttime=love_wave_time, type="relative")