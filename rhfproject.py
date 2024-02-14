import numpy as np
from scipy import linalg as la
import itertools

#Finding Vnu
def VNU():
    with open('geometry.dat', 'r') as file:
        content=file.readlines()
    xyzlist=[]
    for (i,line) in enumerate(content):
        if i > 1:
            xyz=list(filter(None, line.split()))
            xyzlist.append(xyz)
    pairs=list(itertools.combinations(xyzlist,2))
        
    atomic_charges = {"H": 1, "He":2, "Li": 3, "Be":4, "B":5, "C":6, "N":7, "O":8}

    updated_pairs=[]
    for pair in pairs:
        updated_pair=[]
        for element in pair:
            if element[0] in atomic_charges:
                updated_pair.append([atomic_charges[element[0]]]+element[1:])
            else:
                updated_pair.append(element)
        updated_pairs.append(updated_pair)
    float_pairs=[]
    for pairs in updated_pairs:
        float_pair=[]
        for sublist in pairs:
            float_sublist=[]
            for element in sublist:
                float_sublist.append(float(element))
            float_pair.append(float_sublist)
        float_pairs.append(float_pair)
    Vnu=0
    for pair in float_pairs:
        part=(pair[0][0]*pair[1][0])/(np.sqrt((pair[0][1]-pair[1][1])**2+(pair[0][2]-pair[1][2])**2+(pair[0][3]-pair[1][3])**2))
        Vnu +=part
    return Vnu


#forming the orthogonalizer
def orthogonalizer():
    filepaths="s.dat"
    columns=np.loadtxt(filepaths, dtype=float)
    smatrix=np.zeros([24,24])
    for row, col, value in columns:
        smatrix[int(row-1),int(col-1)]=value
    eigenvalues, eigenvectors = la.eigh(smatrix)
    e=np.zeros([len(eigenvalues), len(eigenvalues)])
    inve=np.zeros([len(eigenvalues),len(eigenvalues)])
    for i in range(len(eigenvalues)):
        inve[i,i]=1/(np.sqrt(eigenvalues[i]))
    invsqrtsmatrix=eigenvectors@inve@eigenvectors.T
    return invsqrtsmatrix

filepathhcore="hcore.dat"
columnhcore=np.loadtxt(filepathhcore, dtype=float)
#core Hamiltonian matrix elements in AO basis onto a 2 index tensor
hcore=np.zeros([24,24])
for mu, nu, value in columnhcore:
    hcore[int(mu-1), int(nu-1)]=value

filepatheri="eri.dat"
columneri=np.loadtxt(filepatheri, dtype=float)
#eri onto a 4 index tensor
eri=np.zeros([24,24,24,24])
for i, j, k, l, value in columneri: 
    eri[int(i-1), int(j-1), int(k-1), int(l-1)]=value


def fockmatrix(D):
    #getting the v matrix
    vmatrix=np.zeros([eri.shape[0],eri.shape[2]])
    for i in range(eri.shape[0]):
        for j in range(eri.shape[1]):
            for k in range(eri.shape[2]):
                for l in range(eri.shape[3]):
                    vmatrix[i,k]+=(eri[i, j, k, l]-(1/2)*eri[i,j,l,k])* D[l,j]
    #print("vmatrix:",vmatrix)

    fockmatrix=hcore+vmatrix
    #print("fockmatrix:", fockmatrix)
    return (hcore, vmatrix, fockmatrix)


#orthogonalizing fock matrix
def fockortho(invsqrtsmatrix, fockmatrix):
    #Transforming fock matrix to orthogonalized AO basis 
    ftildematrix=np.matmul(np.matmul(invsqrtsmatrix,fockmatrix), invsqrtsmatrix)
    #print("ftildemat:", ftildematrix)
    return ftildematrix

#diagonalizing orthogonalized fock matrix to get e and MO coefficients C
def diagfockortho(ftildematrix):
    #Diagonalizing ftildematrix
    orbe, Ctilde=la.eigh(ftildematrix) 
    return Ctilde 


#back transforming C to original AO basis 
def MOcoeff(invsqrtsmatrix, Ctilde):
    C=np.matmul(invsqrtsmatrix,Ctilde)
    #print("C:", C)
    return C

#building the new density matrix
def densitymatrix(C):
    n_occ=5 #number of occupied orbitals, can change based on molecule
    Den=np.zeros([C.shape[0],C.shape[0]])
    for i in range(n_occ):
        for nu in range(C.shape[0]):
             for mu in range(C.shape[0]):
                Den[mu,nu]+=C[mu,i]*C[nu,i]
    Dnew=2.*Den
    return Dnew

#finding the energy
def energy(D, vnu, invsqrtsmatrix):
    (hcore, vmatrix, fockmatrix_result)=fockmatrix(D)
    ftildematrix=fockortho(invsqrtsmatrix, fockmatrix_result)
    Ctilde=diagfockortho(ftildematrix)
    C=MOcoeff(invsqrtsmatrix, Ctilde)
    Dnew=densitymatrix(C)
    #print("Dnew: ", Dnew)
    Ee=0
    for mu in range(hcore.shape[0]):
        for nu in range(hcore.shape[1]):
            Ee+=(hcore[mu,nu]+(1/2)*vmatrix[mu,nu])*Dnew[nu,mu]
 
    E=vnu+Ee
    return (E,Dnew)

#this is the main loop going through the iterations 
if __name__ == '__main__':
    vnu=9.0627127677 #This is a test value, but you can replace with VNU()
    invsqrtsmatrix=orthogonalizer()
    D = np.zeros([24,24])
    previous_E = np.infty
    iteration = 0
    while iteration<100: 
        (E,Dnew) = energy(D, vnu, invsqrtsmatrix)
        if abs(E-previous_E)<1e-8:
            print(E)
            break
        else:
            print(f'i = {iteration}, error = {abs(E-previous_E)}')
            previous_E=E
            print(E)
            D=Dnew
        iteration = iteration + 1

