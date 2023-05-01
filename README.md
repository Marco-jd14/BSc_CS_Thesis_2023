# BSc_CS_Thesis_2023

For this project, I got handed a local .sql file created with MySQL.
This file contained the commands that had to be executed in order to create a local SQL database from scratch, and then fill the database with data.

## Installing MySQL
If you do not have MySQL already installed, you can download it [here](https://dev.mysql.com/downloads/installer/). 
Choose your operating system, and then your preferred version of MySQL. 
MySQL then prompts you to login or make a new account, but you can choose to immediately download using a button near the bottom of the page.
Run the installer, and follow the instructions on screen, or follow a tutorial such as
https://www.javatpoint.com/how-to-install-mysql

## Connecting to MySQL from Python
Once you have verified a working version of MySQL on your system, it is now possible to connect to it from Python.
Make sure you have the Python package `mysql` installed by running the command\
`pip install mysql-connector-python`

Then, to initiate the database using the commands from the .sql file, run the script `init_db_from_file.py` provided in the `database` folder from this repository.
This should successfully create and fill a database from scratch, making it ready to be queried from other scripts.
