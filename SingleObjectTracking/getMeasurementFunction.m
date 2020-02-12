function [H, J] = getMeasurementFunction(pattern, quatMotionType, motionType, name)
%GETMEASUREMENTFUNCTION Construct the measurement function from a kalman
%filter. 
%
%   The measurement function is commonly denoted by H and maps from the
%   state space to the measurement space, i.e. z_t = H_t(x_t) + v_t.
%
%   This function (so far) only works for the specific problem at hand,
%   that is the state is 14-dimensional and stores next to velocity
%   information the position and the rotation (stored in quaternion form).
%   Since there are several markers in one pattern H maps a position to 4
%   positions which are offset by the markers in the rotated pattern.
%
%   Furthermore this function also precomputes the jacobian, J, of H, which
%   is used as a linear approximation to the actual H in the extended
%   kalman filter.

    % Get a function handle which produces the corresponding rotation matrix when supplied
    % with a quaternion.   
    if strcmp(quatMotionType, 'brownian')
        if strcmp(motionType, 'constAcc')
            % Work woth symbolic variables in order to compute the jacobian
            % automatically.
            syms x y z vx vy vz ax ay az q1 q2 q3 q4 real;

            % Mapping from state space to measurement space
            H = reshape( (Rot([q1;q2;q3;q4]) * pattern')' + [x; y; z]', [], 1 );

            % Let the Symbolic Math Toolbox calculate the jacobian.
            % The transform back to a regular matlab function.
            J = matlabFunction( jacobian(H, [x; y; z; vx; vy; vz; ax; ay; az; q1; q2; q3; q4])) ;
            %J = matlabFunction( jacobian(H, [x; y; z; vx; vy; vz; q1; q2; q3; q4]), 'File', ['tracking/jacobian/' name] ) ;

            H = @(xvec) reshape( (Rot(xvec(3*3+1:3*3+4)) * pattern')' + xvec(1:3)', [], 1 );
        else
            % Work woth symbolic variables in order to compute the jacobian
            % automatically.
            syms x y z vx vy vz q1 q2 q3 q4 real;

            % Mapping from state space to measurement space
            H = reshape( (Rot([q1;q2;q3;q4]) * pattern')' + [x; y; z]', [], 1 );

            % Let the Symbolic Math Toolbox calculate the jacobian.
            % The transform back to a regular matlab function.
            J = matlabFunction( jacobian(H, [x; y; z; vx; vy; vz; q1; q2; q3; q4])) ;
            %J = matlabFunction( jacobian(H, [x; y; z; vx; vy; vz; q1; q2; q3; q4]), 'File', ['tracking/jacobian/' name] ) ;

            H = @(xvec) reshape( (Rot(xvec(2*3+1:2*3+4)) * pattern')' + xvec(1:3)', [], 1 );
        end
    else
        if strcmp(motionType, 'constAcc')
            %TODO
            fprintf('notimplementedyet')
        else
            % Work woth symbolic variables in order to compute the jacobian
            % automatically.
            syms x y z vx vy vz q1 q2 q3 q4 vq1 vq2 vq3 vq4 real;

            % Mapping from state space to measurement space
            H = reshape( (Rot([q1;q2;q3;q4]) * pattern')' + [x; y; z]', [], 1 );

            % Let the Symbolic Math Toolbox calculate the jacobian.
            % The transform back to a regular matlab function.
            J = matlabFunction( jacobian(H, [x; y; z; vx; vy; vz; q1; q2; q3; q4; vq1; vq2; vq3; vq4]) ) ;
            %J = matlabFunction( jacobian(H, [x; y; z; vx; vy; vz; q1; q2; q3; q4; vq1; vq2; vq3; vq4]), 'File', ['tracking/jacobian/' name] ) ;

            H = @(x) reshape( (Rot(x(2*3+1:2*3+4)) * pattern')' + x(1:3)', [], 1 );
        end
    end
        
end

