##############################################
# Programmer: Aiden Tabrah
# Class: CptS 322-01, Fall 2025
# Programming Assignment #7
# 11/4/25
# 
# Description: This program computes creates a class that can
# automatically grab data from a csv file, do some data 
# cleanup, compute statistics, and do outer/inner joins.
##############################################

import copy
import csv
from tabulate import tabulate
from mysklearn import myutils

class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names (list of str): M column names
        data (list of list of obj): 2D data structure storing mixed type data.
            There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Parameters:
            column_names (list of str): initial M column names (None if empty)
            data (list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    def pretty_print(self):
        """Prints the table in a nicely formatted grid structure."""
        print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            tuple: (N, M) where N is number of rows and M is number of columns
        """
        return (len(self.data), len(self.data[0]))

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Parameters:
            col_identifier (str or int): string for a column name or int
                for a column index
            include_missing_values (bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Raises:
            ValueError: if col_identifier is invalid
        """
        
        column_values = []
        column_index = 0
        
        if isinstance(col_identifier, int):
            if col_identifier <= len(self.column_names):
                raise ValueError()
            column_index = col_identifier
        else:
            if self.column_names.count(col_identifier) == 0:
                raise ValueError()
            column_index = self.column_names.index(col_identifier)

        for row in self.data:
            value = row[column_index]
            if value == None or value == "" or value == "NA":
                if include_missing_values:
                    value = "NA" 
                else:
                    continue
            column_values.append(value)
                    
        return column_values

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leaves values as-is that cannot be converted to numeric.
        """
        
        for row in self.data:
            for i in range(len(self.column_names)):
                try:
                    row[i] = float(row[i])
                except:
                    pass

    def drop_rows(self, row_indexes_to_drop):
        """Remove rows from the table data.

        Parameters:
            row_indexes_to_drop (list of int): list of row indexes to remove from the table data.
        """
        popped = 0
        for drop_index in row_indexes_to_drop:
            self.data.pop(drop_index - popped)
            popped += 1

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Parameters:
            filename (str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: returns self so the caller can write code like
                table = MyPyTable().load_from_file(fname)

        Notes:
            Uses the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load.
        """
        
        with open(filename, 'r') as csvfile:
            reader = csv.reader(csvfile)
            self.column_names = next(reader)
            for row in reader:
                self.data.append(row)
            
        self.convert_to_numeric()
        
        return self

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Parameters:
            filename (str): relative path for the CSV file to save the contents to.

        Notes:
            Uses the csv module.
        """
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.column_names)
            writer.writerows(self.data)

    def find_duplicates(self, key_column_names):
        """Returns a list of indexes representing duplicate rows.
        Rows are identified uniquely based on key_column_names.

        Parameters:
            key_column_names (list of str): column names to use as row keys.

        Returns:
            list of int: list of indexes of duplicate rows found

        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s).
            The first instance of a row is not considered a duplicate.
        """
        
        key_indexes = [self.column_names.index(k) for k in key_column_names]
        
        duplicated_indexes = []
        
        for i in range(len(self.data)):
            duplicate = False
            
            for j in range(0, i):
                rows_match = True

                for column_index in key_indexes:
                    if self.data[i][column_index] != self.data[j][column_index]:
                        rows_match = False
                        break
                    
                if rows_match:
                    duplicate = True
                    break

            if duplicate:
                duplicated_indexes.append(i)
        
        return duplicated_indexes

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA")."""
        popped = 0
        
        for i in range(len(self.data)):
            popped_index = i - popped
            for entry in self.data[popped_index]:
                if entry == "NA":
                    self.data.pop(popped_index)
                    popped += 1
                    break
                

    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column
        by the column's original average.

        Parameters:
            col_name (str): name of column to fill with the original average (of the column).
        """
        column_index = self.column_names.index(col_name)
        
        sum = 0
        entries = 0
        for row in self.data:
            if row[column_index] != "NA":
                sum += row[column_index]
                entries += 1
                
        average = sum / entries
        for row in self.data:
            if row[column_index] == "NA":
                row[column_index] = average

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.
            min: minimum of the column
            max: maximum of the column
            mid: mid-value (AKA mid-range) of the column
            avg: mean of the column
            median: median of the column

        Parameters:
            col_names (list of str): names of the numeric columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]

        Notes:
            Missing values in the columns to compute summary stats
            should be ignored.
            Assumes col_names only contains the names of columns with numeric data.
        """
        new_py_table = MyPyTable(data=[])
        new_py_table.column_names = ["attribute", "min", "max", "mid", "avg", "median"]
        
        for column in col_names:
            column_data = self.get_column(column, False)
            
            if len(column_data) == 0:
                continue
            
            min_value = min(column_data)
            max_value = max(column_data)
            avg = sum(column_data) / len(column_data)
            mid = (max_value + min_value) / 2
            
            column_data.sort()
            
            # Median calculation
            if len(column_data) % 2 == 0:
                halfway = len(column_data) // 2
                median = (column_data[halfway - 1] + column_data[halfway] ) / 2
            else:
                midpoint = len(column_data) // 2
                median = column_data[midpoint]
            
            new_py_table.data.append([column, min_value, max_value, mid, avg, median])
            
        return new_py_table

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined
        with other_table based on key_column_names.

        Parameters:
            other_table (MyPyTable): the second table to join this table with.
            key_column_names (list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        
        new_table = MyPyTable()
        headers = self.column_names.copy()
        for column in other_table.column_names:
            if headers.count(column) == 0:
                headers.append(column)
        new_table.column_names = headers
        
        for _, row in enumerate(self.data):
            for _, other_row in enumerate(other_table.data):
                rows_match = True
                
                for column in key_column_names:
                    column_index = self.column_names.index(column)
                    other_column_index = other_table.column_names.index(column)
                    
                    if row[column_index] != other_row[other_column_index]:
                        rows_match = False
                        break
                    
                if rows_match:
                    new_entry = []
                    for column in new_table.column_names:
                        if self.column_names.count(column) > 0:
                            new_entry.append(row[self.column_names.index(column)])
                        else:
                            new_entry.append(other_row[other_table.column_names.index(column)])
                    new_table.data.append(new_entry)
        return new_table

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with
        other_table based on key_column_names.

        Parameters:
            other_table (MyPyTable): the second table to join this table with.
            key_column_names (list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pads attributes with missing values with "NA".
        """
        
        new_table = MyPyTable()
        headers = self.column_names.copy()
        for column in other_table.column_names:
            if headers.count(column) == 0:
                headers.append(column)
        new_table.column_names = headers
        
        matched_left  = [False] * len(self.data)
        matched_right = [False] * len(other_table.data)
        
        for i, row in enumerate(self.data):
            for j, other_row in enumerate(other_table.data):
                rows_match = True
                
                for column in key_column_names:
                    column_index = self.column_names.index(column)
                    other_column_index = other_table.column_names.index(column)
                    
                    if row[column_index] != other_row[other_column_index]:
                        rows_match = False
                        break
                    
                if rows_match:
                    new_entry = []
                    for column in new_table.column_names:
                        if self.column_names.count(column) > 0:
                            new_entry.append(row[self.column_names.index(column)])
                        else:
                            new_entry.append(other_row[other_table.column_names.index(column)])
                    new_table.data.append(new_entry)
                    matched_left[i] = True
                    matched_right[j] = True
        
        # LEFT-ONLY rows
        for i, row in enumerate(self.data):
            if not matched_left[i]:
                new_entry = []
                for column in new_table.column_names:
                    if column in self.column_names:
                        new_entry.append(row[self.column_names.index(column)])
                    else:
                        new_entry.append("NA")
                new_table.data.append(new_entry)

        # RIGHT-ONLY rows
        for j, other_row in enumerate(other_table.data):
            if not matched_right[j]:
                new_entry = []
                for column in new_table.column_names:
                    if column in other_table.column_names:
                        new_entry.append(other_row[other_table.column_names.index(column)])
                    else:
                        new_entry.append("NA")
                new_table.data.append(new_entry)
        return new_table
    
    def select_data(self, row_indexes):
        """Gets a table's specified rows and turns it into it's own table.
        
        Parameters:
            row_indexes (list of int): instance indexes to copy from the table.
            
        Returns:
            selected data (list of int): instances
        """
        
        selected_data = MyPyTable()
        selected_data.column_names = self.column_names.copy()
        
        for i in row_indexes:
            selected_data.data.append(self.data[i])
        
        return selected_data
    
    def create_frequency_diagram(self, column_name):
        """Creates a frequency diagram based off of a categorical attribute

        Parameters:
            column_name (str): string for a column name
            
        Notes:
            Displays the diagram
        """
        column = self.get_column(column_name)
        data_dictionary = myutils.get_frequency(column)

        myutils.frequency_diagram(data_dictionary, column_name, "cars")

    def create_histogram(self, column_name):
        """Creates a histogram based off of numerical attribute.

        Args:
            column_name (str): String for a column name
        
        Notes:
            Displays the diagram
        """
        
        column = self.get_column(column_name)
        myutils.histogram_diagram(column, column_name)
    
    def normalize_values(self, column_names):
        """Returns a new table with normalized values for the selected columns

        Parameters:
            column_names (list of str): Columns to normalize

        Returns:
            MyPyTable: A new Pytable that is the same as the old, but with normalized values for the columns specified
        """
        new_table = MyPyTable()
        new_table.column_names = self.column_names.copy()
        new_table.data = [row.copy() for row in self.data]
            
        for name in column_names:
            i = self.column_names.index(name)
            
            column = [row[i] for row in new_table.data]
            max_value = max(column)
            min_value = min(column)
            
            for j, value in enumerate(column):
                new_table.data[j][i] = (value - min_value) / (max_value - min_value)
        
        return new_table
    
    def select_columns(self, column_names):
        """Returns a new table with just the selected columns

        Parameters:
            column_names (list of str): Columns to select

        Returns:
            MyPyTable: A new Pytable that is the same as the old, just with only the selected columns
        """
        new_table = MyPyTable()
        new_table.column_names = column_names
        
        for row in self.data:
            instance = []
            
            for name in column_names:
                i = self.column_names.index(name)
                instance.append(row[i])
            
            new_table.data.append(instance)
        
        return new_table
    
    def replace_column(self, column_name, new_values):
        """Replace the values of an existing column.

        Parameters:
            column_name (str): name of the column to replace
            new_values (list): new values, must match number of rows
        """
        if column_name not in self.column_names:
            raise ValueError(f"Unknown column: {column_name}")

        if len(new_values) != len(self.data):
            raise ValueError("new_values length must match number of rows")

        column_index = self.column_names.index(column_name)
        for i, row in enumerate(self.data):
            row[column_index] = new_values[i]