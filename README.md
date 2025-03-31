<h1>Image-Based Visual Servoing Controller</h1>
An implementation of an image-based visual servoing controller, as described in https://ieeexplore.ieee.org/document/4015997. Developed by Jay Rana (contact: jay.rana1@gmail.com).

<h2>Quick Start</h2>
<h3>Camera Frame</h3>
The camera frame is assumed to be: origin at the center of the image, +x goes to the right, +y goes downwards, and +z goes into the image. All velocities are given relative to this frame (e.g. a positive x velocity means to move right).

<h3>Point Format</h3>
Each point is a tuple of three floats.<br>The first value in the tuple will be the normalized x coordinate of the point, and should have a value in the range (-1.0, 1.0).<br>The second value in the tuple will be the normalized y coordinate of the point, and should have a value in the range (-1.0, 1.0).<br>The third value in the tuple will be the depth of the point in meters, and should have a value greater than 0.0.

<h3>Control Modes</h3>
Several control modes are supported (note: this follows the camera frame noted above):
<ul>
    <li>Two degrees of freedom: x velocity and z velocity (<code>control_mode='2xz'</code>)</li>
    <li>Two degrees of freedom: z velocity and y angular velocity (<code>control_mode='2zy'</code>)</li>
    <li>Four degrees of freedom: x velocity, y velocity, z velocity, and y angular velocity (<code>control_mode='4xyzy'</code>)</li>
</ul>

<h3>Interaction Modes</h3>
Several interaction matrix modes are supported:
<ul>
    <li>Only use the current positions of each point in the error interaction matrix estimate (<code>interaction_mode='curr'</code>)</li>
    <li>Only use the desired positions of each point in the error interaction matrix estimate (<code>interaction_mode='desired'</code>)</li>
    <li>Use the mean of the error interaction matrix estimates from the current and desired positions (<code>interaction_mode='mean'</code>)</li>
</ul>

<h3>Controller Loop</h3>
<ol>
    <li>Instantiate the controller with a control mode (listed above), interaction mode (listed above), and the number of points you will supply to the controller each iteration (must be greater than 0):<br><code>controller = IBVS_Controller(control_mode='2xz', interaction_mode='curr', num_pts=2)</code></li>
    <li>Set the lambda matrix of the controller with a Python list:<br><code>controller.set_lambda_matrix(lambdas=[2.0, 5.0])</code></li>
    <li>Set the desired positions of each of your points in the image:<br><code>controller.set_desired_points(curr_pts=[(-0.5, -0.5, 1.0), (0.5, 0.5, 1.0)])</code></li>
    <li>For each loop of the iteration:</li>
    <ol type="a">
        <li>Set the current positions of each of your points in the image:<br><code>controller.set_current_points(curr_pts=[(-0.2, -0.2, 5.0), (0.2, 0.2, 5.0)])</code></li>
        <li>Check if the error is within some threshold, e.g.:<br><code>if np.linalg.norm(controller.errs) < 0.1:<br>break</code></li>
        <li>Calculate the interaction matrix of the control for this iteration:<br><code>controller.calculate_interaction_matrix()</code></li>
        <li>Calculate the output velocities and save them to a variable:<br><code>vels = controller.calculate_velocities()</code></li>
        <li>Apply the output velocities to your motor controllers (note: your robot may have a different frame than your camera!)
    </ol>
</ol>

<h2>Implementation Details</h2>   
The general control equation is: `vels = -1 * lambda_matrix * L_e_est_pinv * errs`, where `vels` is the vector of output velocities, `lambda_matrix` is the diagonal scaling matrix, `L_e_est_pinv` is the Moore-Penrose pseudoinverse of the error interaction matrix estimate, and `errs` is the vector of errors between the current and desired points. `vels` has dimensions `d x 1`. `lambda_matrix` has dimensions `d x d`. `L_e_pinv` has dimensions `d x 2p`. `errs` has dimensions `2p x 1`. `d` denotes the number of degrees of freedom of the controller. `p` denotes the number of points that will be supplied to the controller.

To instantiate the controller, call `IBVS_Controller()` with the chosen control mode, interaction mode, and the number of points that will be supplied to the controller. Each point should be a tuple of 3 floats. The first value in the tuple will be the normalized x coordinate of the point, and should have a value in the range (-1.0, 1.0). The second value in the tuple will be the normalized y coordinate of the point, and should have a value in the range (-1.0, 1.0). The third value in the tuple will be the depth of the point in meters, and should have a value greater than 0.0.

To set the lambda matrix, call `set_lambda_matrix()` with the list of lambda scalars. The list of scalars should have the same length as the number of degrees of freedom of the controller. The list will become a diagonal matrix.

To calculate `L_e_est_pinv`, call `calculate_interaction_matrix()` after setting the current/desired points. Each point has the same format as in `IBVS_Controller()`. If you are using the `curr` interaction mode, `L_e_est_pinv` will be equal to `L_e_pinv`, which is the Moore-Penrose pseudoinverse of the current error interaction matrix. If you are using the `desired` interaction mode, `L_e_est_pinv` will be equal to `L_e_desired_pinv`, which is the Moore-Penrose pseudoinverse of the desired error interaction matrix. If you are using the `mean` interaction mode, `L_e_est_pinv` will be equal to `0.5 * pinv(L_e + L_e_desired)`, where `L_e` is the current error interaction matrix and `L_e_desired` is the desired error interaction matrix.

To set the current positions for each point, call `set_current_points()` with the list of current points. Each point has the same format as in `IBVS_Controller()`. If you are using the `desired` interaction mode, the current depth (third value in each tuple) will be ignored and can be safely set to `None`. If you are using another interaction mode, you must specify the desired depth of each point.

To set the desired positions for each point, call `set_desired_points()` with the list of desired points. Each point has the same format as in `IBVS_Controller()`. If you are using the `curr` interaction mode, the desired depth (third value in each tuple) will be ignored and can be safely set to `None`. If you are using another interaction mode, you must specify the desired depth of each point.

If both the current and desired points have been defined or updated, the controller will automatically calculate the error vector with the `calculate_error_vector()` function.

Once the lambda matrix, `L_e_est_pinv`, and the error vector have all been set/calculated, you can call `calculate_velocities()` to calculate the output velocities and return them as a NumPy array. The velocities will be in the order listed in the associated control mode.
