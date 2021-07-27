import copy
import random
import json
import numpy as np
import sys
import time, itertools
import os, sys, os.path as osp
import argparse, torch
import numpy as np
from torch_geometric.data import Dataset, Data, DataLoader
import torch
# from sklearn.metrics import f1_score, precision_score, recall_score


pyTypesOfStatements = ['body', 'SubscriptStore', 'args', 'CompareEqEq', 'CompareLtELtELt', 'UnaryOpInvert', 'Break', 'ListStore', 'UnaryOpNot', 'AugAssignLShift', 'IfExp', 'Module', 'While', 'CompareNotEqEqNotEq', 'BinOpBitOr', 'Yield', 'defaults', 'AugAssignMult', 'CompareLtELtLt', 'BinOpSub', 'CompareGtELt', 'CompareEqNotEq', 'CompareNotEqEq', 'BinOpPow', 'CompareEqNotEqNotEq', 'ListLoad', 'Global', 'SetComp', 'Continue', 'UnaryOpUAdd', 'NameConstant', 'CompareLtELt', 'BoolOpAnd', 'list', 'CompareGt', 'decorator_list', 'CompareLtLtE', 'Delete', 'Pass', 'arguments', 'Try', 'Return', 'DictComp', 'Assert', 'CompareLtLtELtE', 'CompareNotEqNotEq', 'CompareLtLtELt', 'Name', 'TupleLoad', 'CompareIsNot', 'If', 'BinOpLShift', 'AugAssignAdd', 'CompareEqEqEq', 'CompareIs', 'BinOpDiv', 'finalbody', 'CompareGtEGtEGtE', 'StarredLoad', 'AugAssignBitOr', 'keyword', 'Dict', 'StarredStore', 'AugAssignMod', 'BinOpAdd', 'CompareGtNotEq', 'AugAssignBitXor', 'For', 'attr', 'CompareGtEGtE', 'ExtSlice', 'handlers', 'Set', 'CompareEqEqEqEqEqEqEqEq', 'CompareLtLtLt', 'orelse', 'comprehension', 'AugAssignBitAnd', 'type', 'SubscriptLoad', 'CompareEq', 'withitem', 'CompareGtGtGt', 'Str', 'CompareEqGt', 'BoolOpOr', 'CompareNotEqNotEqNotEq', 'GeneratorExp', 'ListComp', 'AugAssignRShift', 'CompareGtE', 'CompareNotIn', 'CompareLtELtE', 'Expr', 'name', 'FunctionDef', 'BinOpMod', 'Bytes', 'CompareGtEGt', 'Import', 'BinOpBitXor', 'UnaryOpUSub', 'alias', 'AttributeStore', 'BinOpMult', 'CompareLtLt', 'AttributeLoad', 'NameStore', 'ImportFrom', 'BinOpFloorDiv', 'CompareLt', 'NameDel', 'With', 'CompareLtE', 'CompareEqEqEqEqEqEqEqEqEq', 'NameLoad', 'AugAssignDiv', 'CompareGtGt', 'BinOpBitAnd', 'Assign', 'Slice', 'Call', 'ClassDef', 'SubscriptDel', 'CompareIn', 'CompareNotEqGt', 'CompareLtLtLtE', 'Lambda', 'AugAssignSub', 'AugAssignFloorDiv', 'bases', 'Num', 'CompareLtELtELtE', 'BinOpRShift', 'Index', 'Raise', 'ExceptHandler', 'TupleStore', 'identifier', 'CompareNotEq', 'CompareGtGtE', 'arg']
javaTypeofStatements = ['ClassOrInterfaceDeclaration', 'ClassOrInterfaceType', 'ObjectCreationExpr', 'IfStmt', 'Parameter', 'ArrayType', 'ExplicitConstructorInvocationStmt', 'ContinueStmt', 'SuperExpr', 'IntegerLiteralExpr', 'SwitchEntryStmt', 'DoStmt', 'ArrayCreationLevel', 'ForeachStmt', 'ArrayInitializerExpr', 'FieldDeclaration', 'MethodDeclaration', 'BinaryExpr', 'LambdaExpr', 'EmptyStmt', 'SimpleName', 'NameExpr', 'CastExpr', 'ThisExpr', 'LongLiteralExpr', 'ClassExpr', 'BreakStmt', 'TryStmt', 'UnknownType', 'ForStmt', 'AssignExpr', 'InitializerDeclaration', 'BlockStmt', 'VariableDeclarationExpr', 'VoidType', 'CharLiteralExpr', 'PrimitiveType', 'MethodReferenceExpr', 'ReturnStmt', 'AssertStmt', 'InstanceOfExpr', 'TypeParameter', 'ArrayAccessExpr', 'TypeExpr', 'MethodCallExpr', 'BooleanLiteralExpr', 'UnaryExpr', 'Name', 'ConstructorDeclaration', 'EnumConstantDeclaration', 'FieldAccessExpr', 'ExpressionStmt', 'NullLiteralExpr', 'ThrowStmt', 'LabeledStmt', 'WhileStmt', 'SwitchStmt', 'EnumDeclaration', 'StringLiteralExpr', 'CompilationUnit', 'CatchClause', 'ArrayCreationExpr', 'DoubleLiteralExpr', 'LocalClassDeclarationStmt', 'VariableDeclarator', 'ImportDeclaration', 'WildcardType', 'EnclosedExpr', 'ConditionalExpr']
# discuss req of padding?? 
maxLength=max(len(pyTypesOfStatements),len(javaTypeofStatements))

# will give an encoded matrix for setofStatements
def oneHotEncoder(setofStatements,langType):
  jsonArrayConverted = json.loads(setofStatements)
  encodeMatrix = np.zeros((len(jsonArrayConverted),maxLength), dtype = 'int32')

  # check if python or java
  if langType == "python":
    for i in range(len(jsonArrayConverted)):
        idx = pyTypesOfStatements.index(jsonArrayConverted[i]['type'])
        encodeMatrix[i][idx] = 1
  else:
    for i in range(len(jsonArrayConverted)):
        idx = javaTypeofStatements.index(jsonArrayConverted[i]['type'])
        encodeMatrix[i][idx] = 1
  return encodeMatrix

# prepares the adjacency matrix
def adjacencyMatrixCreator(setofStatements):
  jsonArrayConverted = json.loads(setofStatements)
  num_nodes=len(jsonArrayConverted)
  srcArr=[]
  desArr=[]
  for i in range(len(jsonArrayConverted)):
    if('children' in jsonArrayConverted[i]):
      listOfChildren = jsonArrayConverted[i]['children']
      parentId= int(jsonArrayConverted[i]['id'])
      
      # directed edges / undirected edges ??
      for child in listOfChildren:
        childId = int(child)
        srcArr.append(parentId)
        desArr.append(childId)
  nsrcArr = np.array(srcArr).astype('int32')
  ndesArr = np.array(desArr).astype('int32')
  adjacencyMatrix = np.row_stack((nsrcArr,ndesArr))
  return adjacencyMatrix,num_nodes

