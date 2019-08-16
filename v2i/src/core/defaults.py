DEFAULT_DICT = {

    # Simulation Settings
    't-period': 0.1, # in secs
    'seed' : 1,
    'local-view' : 10, # in metres
    'extended-view': 20, # in metres
    'cell-size' : 1, # in metres
    'reg-size': 5, # size of 1 comm region in metres

    # Reward Function
    'collision-penalty': -10,
    'comm-penalty': 0.0,

    # IDM Settings
    'max-speed': 9, # in m/s

    # UI settings
    'render': True,
    'fps': 120,

    # Frame Skip
    'frame-skip-value': 1,

    # Traffic Lights
    'enable-tf' : True,
}
