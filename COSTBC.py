import numpy as np
# implementation of algorithm found in A Systematic Design of High-Rate Complex
# Orthogonal Space-Time Block Codes by Weifeng Su
#
# implementation by Markos Loizou

class COSTBC:
    def __init__(self,nt):
        self.nt = nt
        self.construct()
    
    def construct(self):
        #initialize
        self.matrix = self.strIdentity(1,'1')
        self.symbols_used = 1
        self.row_count = 1
        self.iteration = 1
        
        #generate matrices G2 to Gn
        for i in range(self.nt-1):
            self.countRows()
            s = self.matrix.shape
            self.row_count = s[0]
            self.matrix_old = self.matrix
            self.matrix = np.ndarray(shape =(s[0],s[1]+1), dtype=object)
            self.matrix[:s[0], :s[1]] = self.matrix_old
            self.fillColumn(s[1])
            self.fillElements()
            self.orthogonalizeColumns()
            self.addRows()
            self.iteration += 1
            self.symbols_used += max(self.conj_rows,self.nconj_rows)
        
    def strIdentity(self, n, string):
        eye = np.ndarray(shape=(n,n), dtype=object)
        for i in range(0,n):
            eye[i,i] = string
        return eye
    
    def countRows(self):
        self.nconj_rows = 0
        self.conj_rows = 0 
        self.conj_rows_list = []
        self.nconj_rows_list = []
        s = self.matrix.shape
        
        for i in range(s[0]):
            nconj = True
            conj = True
            
            for j in range(s[1]):
                if(self.matrix[i,j] != '0'):
                    if('*' in self.matrix[i,j]):
                        nconj = False
                    else:
                        conj = False
            
            if(nconj == True):
                self.conj_rows += 1
                self.conj_rows_list.append(i)
            if(nconj == True):
                self.nconj_rows += 1
                self.nconj_rows_list.append(i)
    
    #step 2 of the algorithm
    def fillColumn(self, n):
        if(self.nconj_rows >= self.conj_rows):
            count = 1
            
            for index in self.nconj_rows_list:
                self.matrix[index,n] = str(self.symbols_used + count)
                count += 1
        elif(self.nconj_rows >= self.conj_rows):
            count = 1
            
            for index in self.conj_rows_list:
                self.matrix[index,n] = str(self.symbols_used + count) + str('*')
                count += 1
    
    #step 3 of the algorithm
    def fillElements(self):
        matrix_tmp = self.matrix
        s = self.matrix.shape
        
        self.matrix = np.ndarray(shape=(s[0] + self.symbols_used, s[1]),dtype=object)
        self.matrix[:s[0], :s[1]] = matrix_tmp
        
        if(self.nconj_rows <= self.conj_rows):
            for i  in range(self.symbols_used):
                self.matrix[s[0] + i,s[1]-1] = str(i+1) + str('*')
        else:
            for i  in range(self.symbols_used):
                self.matrix[s[0] + i,s[1]-1] = str(i+1)
    
    #step 4 of the algorithm
    def orthogonalizeColumns(self):
        s = self.matrix_old.shape
        for row in range(s[0]):
            for column in range(s[1]):
                for i in range(s[1]):
                    if(self.matrix_old[row,column].find(str(i + 1)) != -1):
                        if(self.matrix_old[row,column].find('-') != -1):
                            sym = self.matrix[self.row_count + i,column]
                            
                            self.matrix[self.row_count + i, column] = self.SymbolConjugate(sym) #had + 1 for iteration bellow as well
                            
                        else:
                            sym = self.matrix[self.row_count + i,column]
                            if(sym is None):
                                sym = '0'
                            
                            self.matrix[self.row_count + i, column] = self.SymbolNegative(self.SymbolConjugate(sym))
        self.fillEmpty()
    
    
    
    #used in part 4 of the algorithm to set empty entries to zero
    def fillEmpty(self):
        s = self.matrix.shape
        
        for row in range(s[0]):
            for column in range(s[1]):
                if(not self.matrix[row,column]):
                    self.matrix[row,column] = str(0)


    #final step of the algorithm 
    def addRows(self):
        s = self.matrix_old.shape
        
        RowCount = s[0] + self.symbols_used
        for row in range(s[0], RowCount):            
            for col1 in range(self.iteration):
                for col2 in range(col1+1,self.iteration):
                    NeedNewRow = False
                    if (self.matrix[row,col1] != str(0) and self.matrix[row,col2] != str(0)):
                        NeedNewRow = True
                        X11 = self.matrix[row,col1]
                        X12 = self.matrix[row,col2]
                        
                        for newRow in range(s[0] + self.symbols_used, RowCount):
                            X21 = self.matrix[newRow,col1]
                            X22 = self.matrix[newRow,col2]
                            
                            if(X21 == '0' and self.SymbolAbsolute(X22) == self.SymbolAbsolute(X11)):
                                if(X11 == self.SymbolConjugate(X22)):
                                    self.matrix[newRow,col1] = self.SymbolNegative(self.SymbolConjugate(X12))
                                if(X11 == self.SymbolNegative(self.SymbolConjugate(X22))):
                                    self.matrix[newRow,col1] = self.SymbolConjugate(X12)
                            if(X22 == '0' and self.SymbolAbsolute(X12) == self.SymbolAbsolute(X21)):
                                if(X21 == self.SymbolConjugate(X12)):
                                    self.matrix[newRow,col2] = self.SymbolNegative(self.SymbolConjugate(X11))
                                if(X21 == self.SymbolNegative(self.SymbolConjugate(X12))):
                                    self.matrix[newRow,col1] = self.SymbolConjugate(X11)
                                    NeedNewRow = False
                                
                        if(NeedNewRow == True):
                            RowCount += 1
                            s = self.matrix.shape
                            while(s[0]<= RowCount):
                                self.AddEmptyRow()
                                s = self.matrix.shape
                            
                            self.matrix[RowCount,col1] = self.SymbolNegative(self.SymbolConjugate(self.matrix[row,col2]))
                            self.matrix[RowCount,col2] = self.SymbolConjugate(self.matrix[row,col1])
            
            
    def AddEmptyRow(self,n=1):
        tmp_matrix = self.matrix
        s = self.matrix.shape
        self.matrix = np.ndarray(shape=(s[0]+n, s[1]), dtype=object)
        self.matrix[:s[0],:s[1]] = tmp_matrix
        self.matrix[s[0]:,:] = '0'
        
    
    def SymbolOnly(self,s):
        x = s
        f = x.find('-')
        #delete negative sign
        if(f != -1):
            x = x[:f] + x[f+1:]
            
        f = x.find('*')
        if(f != -1):
            x = x[:f] + x[f+1:]
        
        return x
    
    
    def SymbolAbsolute(self,s):
        return self.SymbolOnly(s)
    
    
    def SymbolConjugate(self,s):
        x = s
        
        f = x.find('*')
        
        if(f != -1):
            x = x[:f] + x[f+1:]
        else:
            x = x + '*'
        
        return x

    def SymbolNegative(self,s):
        x = s
        
        f = x.find('-')
        if(f != -1):
            x = x[:f] + x[f+1:]
        else:
            x = '-' + x
        
        return x






