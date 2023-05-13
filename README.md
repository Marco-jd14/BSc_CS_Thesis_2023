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

Then, to initiate the database using the commands from the .sql file, open the script `init_db_from_file.py` provided in the `database` folder from this repository.
Before running the script, first make sure to update the HOST_ARGS and possibly DATABASE_NAME, which can be found at the top of the script.
If you want to change your password before writing it in the script, you can open the MySQL Command Line Client and enter your original password.
Then, you can run the command `ALTER USER 'root'@'localhost' IDENTIFIED BY 'my_new_password'`.
After making sure the HOST_ARGS are correct, the script should successfully create and fill a database from scratch, making it ready to be queried from other scripts (i.e. `query_db.py`)

## Data prepping
With the database initialized, it is not possible to have an easier look at the data and prepare it for usage.
This is exactly what the script `filter_and_check_data.py` does in the `database` folder.
It filters out irrelevant data, and then further cleans up the data by checking for inconsistencies.
When the script is done, it writes the cleaned-up data back to the database, to new tables.
The filtered data can then be easily retrieved from the database in other scripts.


# Coupon Allocation Algorithms
