function A=fillA(A,EVS,N,k)

% Bestimmung der ersten Spalte von A durch Zeilensummenbedingung */
A(2:k,1)=-sum(A(2:k,2:k),2);

% Bestimmung der ersten Zeile von A durch Maximumsbedingung */
for j=1:k
    A(1,j)=- EVS(1,2:k)*A(2:k,j);
    for l=2:N
        dummy = - EVS(l,2:k)*A(2:k,j);
        if (dummy > A(1,j))
            A(1,j) = dummy;
        end
    end
end

% Reskalierung der Matrix A auf zulaessige Menge */
A=A/sum(A(1,:));

  