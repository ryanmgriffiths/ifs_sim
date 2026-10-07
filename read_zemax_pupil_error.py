"""Save Wavefront Maps for configurations 1–45 as a FITS cube.

Run with OpticStudio's Interactive Extension enabled and Wavefront Map open
at analysis index 10. The cube axes are (configuration, y, x).
"""

import os
from pathlib import Path

import numpy as np
from astropy.io import fits


ANALYSIS_INDEX = 10
FIRST_CONFIGURATION = 1
LAST_CONFIGURATION = 45
OUTPUT_FILE = Path(__file__).resolve().parent / 'wavefront_maps.fits'


def connect_to_opticstudio():
    import clr
    import winreg

    # This boilerplate requires the 'pythonnet' module.
    # The following instructions are for installing the 'pythonnet' module via pip:
    #    1. Ensure you are running a Python version compatible with PythonNET. Check the article "ZOS-API using Python.NET" or
    #    "Getting started with Python" in our knowledge base for more details.
    #    2. Install 'pythonnet' from pip via a command prompt (type 'cmd' from the start menu or press Windows + R and type 'cmd' then enter)
    #
    #        python -m pip install pythonnet

    # determine the Zemax working directory
    aKey = winreg.OpenKey(winreg.ConnectRegistry(None, winreg.HKEY_CURRENT_USER), r"Software\Zemax", 0, winreg.KEY_READ)
    zemaxData = winreg.QueryValueEx(aKey, 'ZemaxRoot')
    NetHelper = os.path.join(os.sep, zemaxData[0], r'ZOS-API\Libraries\ZOSAPI_NetHelper.dll')
    winreg.CloseKey(aKey)

    # add the NetHelper DLL for locating the OpticStudio install folder
    clr.AddReference(NetHelper)
    import ZOSAPI_NetHelper

    pathToInstall = ''
    # uncomment the following line to use a specific instance of the ZOS-API assemblies
    #pathToInstall = r'C:\C:\Program Files\Zemax OpticStudio'

    # connect to OpticStudio
    success = ZOSAPI_NetHelper.ZOSAPI_Initializer.Initialize(pathToInstall);

    zemaxDir = ''
    if success:
        zemaxDir = ZOSAPI_NetHelper.ZOSAPI_Initializer.GetZemaxDirectory();
        print('Found OpticStudio at:', zemaxDir)
    else:
        raise Exception('Cannot find OpticStudio')

    # load the ZOS-API assemblies
    clr.AddReference(os.path.join(os.sep, zemaxDir, r'ZOSAPI.dll'))
    clr.AddReference(os.path.join(os.sep, zemaxDir, r'ZOSAPI_Interfaces.dll'))
    import ZOSAPI

    TheConnection = ZOSAPI.ZOSAPI_Connection()
    if TheConnection is None:
        raise Exception("Unable to intialize NET connection to ZOSAPI")

    TheApplication = TheConnection.ConnectAsExtension(0)
    if TheApplication is None:
        raise Exception("Unable to acquire ZOSAPI application")

    if TheApplication.IsValidLicenseForAPI == False:
        raise Exception("License is not valid for ZOSAPI use.  Make sure you have enabled 'Programming > Interactive Extension' from the OpticStudio GUI.")

    TheSystem = TheApplication.PrimarySystem
    if TheSystem is None:
        raise Exception("Unable to acquire Primary system")

    return TheApplication, TheSystem


def get_wavefront(system):
    wavefront = system.Analyses.Get_AnalysisAtIndex(ANALYSIS_INDEX)
    wavefront.ApplyAndWaitForCompletion()
    results = wavefront.GetResults()
    output = results.GetDataGrid(0)
    return np.asarray(output.Values)


def collect_wavefront_maps(system):
    wavefront_maps = []
    for configuration in range(FIRST_CONFIGURATION, LAST_CONFIGURATION + 1):
        if not system.MCE.SetCurrentConfiguration(configuration):
            raise RuntimeError(f'Could not select configuration {configuration}.')
        wavefront_maps.append(get_wavefront(system))
        print(f'Collected configuration {configuration}')
    return wavefront_maps


if __name__ == '__main__':
    application, system = connect_to_opticstudio()
    wavefront_maps = collect_wavefront_maps(system)
    # Infer y/x dimensions from the maps; require every map to have the same shape.
    stack = np.stack(wavefront_maps, axis=0)
    fits.writeto(OUTPUT_FILE, stack, overwrite=True)
    print(f'Saved {stack.shape} wavefront stack to {OUTPUT_FILE}')
