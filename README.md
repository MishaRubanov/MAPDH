# Multidomain, Automated, Photopatterning of DNA-Functionalized Hydrogels
## Patterning functions for all-in-one scripting for use with **MAPDH**

Multidomain, Automated, Photopatterning of DNA-Functionalized Hydrogels (**MAPDH**) is a platform for the automated fabrication of DNA-functionalized hydrogels using digital, maskless photolithography. The preprint is coming out soon!



## Getting Started

To streamline MAPDH use, example scripts and functions for convenience are provided.

Key files in the main folder:
* Patterning_functions
  * This file contains all necessary functions to run all operations in MAPDH.
* Patterning_example_script
  * This file is an example that uses a few built-in masks to pattern differently shaped hydrogels
* MAPDH_example_script
  * This file is an example for fully automated MADPH - incorporating the flow controller, patterning setup, bright-field and fluorescence imaging into one script for multi-domain patterning and simultaneous imaging.

### Prerequisites

* Install lastest copy of [Micromanager](https://micro-manager.org/)
* Install lastest copy of [Pycromanager](https://github.com/micro-manager/pycro-manager)

* Python and other required packages, with version numbers:
  * Python: 3.8
  * anaconda: 2020.07
  * Scikit-image: 0.19.3
  * Pandas: 1.4.4
  * numpy: 1.18.5
  * PIL: 7.2.0
  * Pycromanager: 0.6.0

### Patterning examples
These are real-time videos for fabrication of hydrogel letters taken using bright-field microscopy. Each letter is fabricated using the message_mask_generator package in Python. Each letter here is the same domain, i.e., each hydrogel has the same composition.

https://user-images.githubusercontent.com/67386551/219878485-338b717b-f008-45a6-b417-a819aa724788.mp4

https://user-images.githubusercontent.com/67386551/219878132-83617d94-9442-419f-b393-dbde8f758fe1.mp4

