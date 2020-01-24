function rotMat = Rot(q)
        q = q/sqrt(sum(q.^2));
        rotMat = eye(3) + 2 * ...
                    [-q(3)^2-q(4)^2         q(2)*q(3)+q(4)*q(1)   q(2)*q(4)-q(3)*q(1);
                      q(2)*q(3)-q(4)*q(1)  -q(2)^2- q(4)^2        q(3)*q(4)+q(2)*q(1);
                      q(2)*q(4)+q(3)*q(1)   q(3)*q(4)-q(2)*q(1)  -q(2)^2-q(3)^2      ;];
        
        
        q_matrix = [q(1) -q(2) -q(3) -q(4);
                    q(2)  q(1) -q(4)  q(3);
                    q(3)  q(4)  q(1) -q(2);
                    q(4) -q(3)  q(2)  q(1)];
        q_bar_matrix = [q(1) -q(2) -q(3) -q(4);
                        q(2)  q(1)  q(4) -q(3);
                        q(3) -q(4)  q(1)  q(2);
                        q(4)  q(3) -q(2)  q(1)];
        product_matrix = q_matrix * q_bar_matrix';
        product_matrix = product_matrix(2:end,2:end);
        rotMat = product_matrix;
end
