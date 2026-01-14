# Sydney Bus Frequency Visualiser

Visit [sudface.github.io/frequency](https://sudface.github.io/frequency) to view the visualiser.

Data is processed from the TfNSW GTFS dataset [here](https://opendata.transport.nsw.gov.au/data/dataset/public-transport-timetables-realtime/resource/9b3bfa13-0053-4008-8575-e30151f05d54)

The processor is in [./getStopFrequency.py](./getStopFrequency.py), and generates a `details.JSON` file for a given date. The main function in use is `getDayServices()`. 

`doWeekdayFreq()` can be used to generate frequency information hardcoded into a GeoJSON file, for visualisation in a GIS software.