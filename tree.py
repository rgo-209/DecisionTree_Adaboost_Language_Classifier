"""
Submitted by: Rahul Golhar(rg1391@rit.edu)
"""
class Root:
    def __init__(self,attributeVal):
        self.true = None
        self.false = None
        self.attributeVal = attributeVal

    def __str__(self):
        return str(self.attributeVal)