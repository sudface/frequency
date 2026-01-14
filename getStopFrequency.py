import pandas as pd
from pprint import pprint
from collections import defaultdict
import numpy as np
DAY_MAX = 60*24

class bcolors:
    PURPLE    = '\033[95m'
    BLUE      = '\033[94m'
    GREEN     = '\033[92m'
    YELLOW    = '\033[93m'
    RED       = '\033[91m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC      = '\033[0m'

def colour(string, *args):
    return "".join(args) + string + bcolors.ENDC

def getInfo(IN_DATE):

    #######################################
    print("Reading Calendar")

    target_date = pd.to_datetime(IN_DATE, format="%Y%m%d")
    weekday = target_date.day_name().lower()   # get day of the input date

    cal = pd.read_csv("calendar.txt", dtype=int)
    cal_dates = pd.read_csv("calendar_dates.txt", dtype=int)

    cal["start_date"] = pd.to_datetime(cal["start_date"], format="%Y%m%d")
    cal["end_date"] = pd.to_datetime(cal["end_date"], format="%Y%m%d")

    # base services from calendar.txt which operate on given date
    base = cal[
        (cal["start_date"] <= target_date) &
        (cal["end_date"] >= target_date) &
        (cal[weekday] == 1)
    ][["service_id"]]

    serviceids = set(base["service_id"])

    # exceptions in calendar_dates
    # Type 2 is exclusion, Type 1 is additional.
    exceptions = cal_dates[cal_dates["date"] == IN_DATE]
    added = exceptions[exceptions["exception_type"] == 1]["service_id"]
    serviceids.update(added)

    removed = exceptions[exceptions["exception_type"] == 2]["service_id"]
    serviceids.difference_update(removed)

    if not serviceids:
        print(f"NO MATCHING SERVICES IN CALENDAR ON DATE {IN_DATE}")
        exit(1)
    ######################################

    ROUTE_BUS_ID = 700

    print("Importing stops")
    stops = pd.read_csv('stops.txt')

    print("Parsing stops")
    stopdata = stops.set_index('stop_id').apply(lambda row: [row['stop_name'], [row['stop_lat'], row['stop_lon']]], axis=1).to_dict()
    # stopdata is now in the format:
    {   
        2155193: ['Withers Rd after Milford Dr', [-33.679708, 150.926346]],
        2155194: ['Withers Rd opp Hills Centenary Park', [-33.681684, 150.928633]],
        2155195: ['Sanctuary, North West Twy', [-33.695491, 150.926742]],
        2155196: ['Commercial Rd after Withers Rd', [-33.68423, 150.930262]],
        2155197: ['Commercial Rd at Caddies Bvd', [-33.686303, 150.924488]],
    }

    ##################################

    print("Importing routes")
    routes = pd.read_csv('routes.txt')
    routes = routes[routes['route_type'] == ROUTE_BUS_ID]

    print("Parsing routes")
    routelist = set(routes['route_id'].tolist())
    # routelist is now a set of route_ids which are not school buses

    routedata = routes.set_index('route_id').apply(lambda row: (row['route_short_name'], row['route_long_name']), axis=1).to_dict()
    # routedata is now in the format {routeid: [route number, route name]}:
    {
        '2501_671': ('671', 'Riverstone to Windsor via McGraths Hill & Vineyard'),
        '2501_672': ('672', 'Windsor to Wisemans Ferry (Loop Service)'),
        '2501_673': ('673', 'Windsor to Penrith via Cranebrook'),
        '2501_674': ('674', 'Windsor to Mount Druitt via South Windsor & Shanes Park'),
    }

    ###################################

    print("Importing trips")
    trips = pd.read_csv('trips.txt')
    tripdata = dict()

    print("Parsing trips")
    for row in trips.itertuples():
        # only include non-schoolbus that operate on given day
        if (row.service_id in serviceids) and (row.route_id in routelist):
            tripdata[row.trip_id] = (row.route_id, row.direction_id)

    tripdata = dict(tripdata)
    # tripdata is now in the format {tripid: [route, direction]}
    {
        "2024057": ('2501_729', 1),
        "2024073": ('2501_728', 0),
    }

    ###################################

    print("Importing stop_times")
    times = pd.read_csv('stop_times.txt')
    times = times[times['trip_id'].isin(tripdata)]

    print("Parsing stop_times")

    # Group the dataframe by stop_id and then collapse it to a list of trip_id and time.
    timedata = times.groupby('stop_id')[['trip_id', 'arrival_time']].apply(lambda x: sorted(zip(x['trip_id'], x['arrival_time']), key=lambda i: i[1])).to_dict()

    # timedata is now in the format {stopid: [(tripid, time)]}
    {
        277079:  [(2024057, '20:25:00'), (2024058, '21:25:00')],
        2770204: [(2024057, '20:28:00'), (2024058, '21:28:00')],
        2770514: [(2024057, '20:29:00'), (2024058, '21:29:00')],
    }

    return [stopdata, routedata, timedata, tripdata]

###################################
getTime = lambda x: int(x[0:2]) * 60 + int(x[3:5])

def doWeekdayFreq(IN_DATE):

    [stopdata, routedata, timedata, tripdata] = getInfo(IN_DATE)

    features = []
    def addFeature(id, data):
        stop = stopdata[id]

        features.append({
            "type": "Feature",
            "properties": {
                "id": id,
                "name": stop[0],
                "am": data[0],
                "pm": data[1],
                "dayBph": data[2],
                "interB": data[3]
            },
            "geometry": {
                "type": "Point",
                "coordinates": [stop[1][1], stop[1][0]]
            }
        })

    #####################
    print("Parsing Gaps")

    getTime = lambda x: int(x[0:2]) * 60 + int(x[3:5])
    avg = lambda x: DAY_MAX if np.isnan(np.mean(x)) else int(np.mean(x))

    for stop, services in timedata.items():
        times = np.sort([getTime(service[1]) for service in services])

        morningPeak = np.diff([i for i in times if getTime("07:00") < i < getTime("08:30")])
        morningPeak = morningPeak[morningPeak != 0]

        arvoPeak    = np.diff([i for i in times if getTime("16:30") < i < getTime("18:30")]) 
        arvoPeak    = arvoPeak[arvoPeak != 0]
        
        morningAvg  = avg(morningPeak)
        arvoAvg     = avg(arvoPeak)

        ##

        dayTime    = np.diff([i for i in times if getTime("07:00") < i < getTime("20:00")]) 
        dayTime    = dayTime[dayTime != 0]
        
        dayBph      = len(dayTime)/13
        dayBph      = np.round(dayBph, 1)
        
        ##
        interTime    = np.diff([i for i in times if getTime("09:30") < i < getTime("14:30")]) 
        interTime    = interTime[interTime != 0]

        interBph      = len(interTime)/5
        interBph      = np.round(interBph, 1)

        addFeature(stop, [morningAvg, arvoAvg, dayBph, interBph])


    #######################################
    print("Saving GeoJSON")

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    import json
    with open("points_weekday.geojson", "w") as f:
        json.dump(geojson, f)

#######################

def doWeekendFreq(IN_DATE):
    [stopdata, routedata, timedata, tripdata] = getInfo(IN_DATE)

    features = []
    def addFeature(id, data):
        stop = stopdata[id]

        features.append({
            "type": "Feature",
            "properties": {
                "id": id,
                "name": stop[0],
                "satBph": data[0],
            },
            "geometry": {
                "type": "Point",
                "coordinates": [stop[1][1], stop[1][0]]
            }
        })


    print("Parsing Gaps SATURDAY")

    getTime = lambda x: int(x[0:2]) * 60 + int(x[3:5])
    avg = lambda x: DAY_MAX if np.isnan(np.mean(x)) else int(np.mean(x))

    for stop, services in timedata.items():
        times = np.sort([getTime(service[1]) for service in services])

        dayTime    = np.diff([i for i in times if getTime("09:00") < i < getTime("19:00")]) 
        dayTime    = dayTime[dayTime != 0]
        
        dayBph      = len(dayTime)/10
        dayBph      = np.round(dayBph, 1)
        
        addFeature(stop, [dayBph])
        
    #######################################
    print("Saving GeoJSON")

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    import json
    with open("pointsSUN.geojson", "w") as f:
        json.dump(geojson, f)

#doWeekendFreq("20260118")

def getDayServices(IN_DATE):

    [stopdata, routedata, timedata, tripdata] = getInfo(IN_DATE)

    data = {
        "routes": routedata,
        "stops": stopdata,
        "trips": tripdata,
        "times": {}
    }

    print("Parsing Gaps")

    for stop, services in timedata.items():
        times = [getTime(service[1]) for service in services]
        tripids = [service[0] for service in services]
        diffs = np.diff(times).tolist()
        data['times'][stop] = [times, tripids, diffs]

    import json
    with open(f"details_{IN_DATE}.json", "w") as f:
        json.dump(data, f)

getDayServices("20260117")