# (C) Jay Rana 2025-03-30

import cv2
import numpy as np
from typing import List, Tuple

class IBVS_Controller():
    '''
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
            <li>Check if the error is within some threshold, e.g.:<br><code>if np.linalg.norm(controller.errs) < 0.1: break</code></li>
            <li>Calculate the interaction matrix of the control for this iteration:<br><code>controller.calculate_interaction_matrix()</code></li>
            <li>Calculate the output velocities and save them to a variable:<br><code>vels = controller.calculate_velocities()</code></li>
            <li>Apply the output velocities to your motor controllers (note: your robot may have a different frame than your camera!)
        </ol>
    </ol>

    <h2>Implementation Details</h2>   
    The general control equation is: <code>vels = -1 * lambda_matrix * L_e_est_pinv * errs</code>, where <code>vels</code> is the vector of output velocities, <code>lambda_matrix</code> is the diagonal scaling matrix, <code>L_e_est_pinv</code> is the Moore-Penrose pseudoinverse of the error interaction matrix estimate, and <code>errs</code> is the vector of errors between the current and desired points. <code>vels</code> has dimensions <code>d x 1</code>. <code>lambda_matrix</code> has dimensions <code>d x d</code>. <code>L_e_pinv</code> has dimensions <code>d x 2p</code>. <code>errs</code> has dimensions <code>2p x 1</code>. <code>d</code> denotes the number of degrees of freedom of the controller. <code>p</code> denotes the number of points that will be supplied to the controller.

    To instantiate the controller, call `IBVS_Controller()` with the chosen control mode, interaction mode, and the number of points that will be supplied to the controller. Each point should be a tuple of 3 floats. The first value in the tuple will be the normalized x coordinate of the point, and should have a value in the range (-1.0, 1.0). The second value in the tuple will be the normalized y coordinate of the point, and should have a value in the range (-1.0, 1.0). The third value in the tuple will be the depth of the point in meters, and should have a value greater than 0.0.

    To set the lambda matrix, call `set_lambda_matrix()` with the list of lambda scalars. The list of scalars should have the same length as the number of degrees of freedom of the controller. The list will become a diagonal matrix.

    To calculate `L_e_est_pinv`, call `calculate_interaction_matrix()` after setting the current/desired points. Each point has the same format as in `IBVS_Controller()`. If you are using the `curr` interaction mode, `L_e_est_pinv` will be equal to `L_e_pinv`, which is the Moore-Penrose pseudoinverse of the current error interaction matrix. If you are using the `desired` interaction mode, `L_e_est_pinv` will be equal to `L_e_desired_pinv`, which is the Moore-Penrose pseudoinverse of the desired error interaction matrix. If you are using the `mean` interaction mode, `L_e_est_pinv` will be equal to `0.5 * pinv(L_e + L_e_desired)`, where `L_e` is the current error interaction matrix and `L_e_desired` is the desired error interaction matrix.
    
    To set the current positions for each point, call `set_current_points()` with the list of current points. Each point has the same format as in `IBVS_Controller()`. If you are using the `desired` interaction mode, the current depth (third value in each tuple) will be ignored and can be safely set to `None`. If you are using another interaction mode, you must specify the desired depth of each point.
    
    To set the desired positions for each point, call `set_desired_points()` with the list of desired points. Each point has the same format as in `IBVS_Controller()`. If you are using the `curr` interaction mode, the desired depth (third value in each tuple) will be ignored and can be safely set to `None`. If you are using another interaction mode, you must specify the desired depth of each point.

    If both the current and desired points have been defined or updated, the controller will automatically calculate the error vector with the `calculate_error_vector()` function.

    Once the lambda matrix, `L_e_est_pinv`, and the error vector have all been set/calculated, you can call `calculate_velocities()` to calculate the output velocities and return them as a NumPy array. The velocities will be in the order listed in the associated control mode.
    '''
    def __init__(self, control_mode: str, interaction_mode: str, num_pts: int):
        assert control_mode == '2xz' or control_mode == '2zy' or control_mode == '4xyzy', f"{control_mode} is not a valid control mode. Please refer to the class docstring to see the list of valid control modes."
        assert interaction_mode == 'curr' or interaction_mode == 'desired' or interaction_mode == 'mean', f"{interaction_mode} is not a valid interaction mode. Please refer to the class docstring to see the list of valid interaction modes."
        assert num_pts > 0, f"{num_pts} is not a valid number of points. You must supply at least one point to the controller each iteration."

        self.control_mode = control_mode

        # two degrees of freedom: x velocity and z velocity
        if self.control_mode == '2xz':
            self.num_degs = 2
        # two degrees of freedom: z velocity and y angular velocity
        elif self.control_mode == '2zy':
            self.num_degs = 2
        # Four degrees of freedom: x velocity, y velocity, z velocity, and y angular velocity
        elif self.control_mode == '4xyzy':
            self.num_degs = 4
        
        self.interaction_mode = interaction_mode
        
        self.vels = None
        self.lambda_matrix = None
        self.L_e_est_pinv = None
        self.errs = None
        
        self.num_pts = num_pts
        self.curr_pts = None
        self.desired_pts = None

    def set_lambda_matrix(self, lambdas: List[float]):
        '''
        Given the list of lambda scalars, set `self.lambda_matrix` to a diagonal matrix from that list.

        :param lambdas: A Python list of lambda scalars. Each scalar will scale one component of the final output velocity. The velocities will be in the order listed in the associated control mode.
        '''
        assert len(lambdas) == self.num_degs, f"You must provide {self.num_degs} lambda scalars in the list."
        self.lambda_matrix = np.diag(lambdas)
    
    def calculate_interaction_matrix(self):
        '''
        Calculate the Moore-Penrose pseudoinverse of the error interaction matrix estimate, `self.L_e_est_pinv`, based on our current control and interaction modes.
        '''
        if self.interaction_mode == 'curr':
            assert self.curr_pts is not None, "You must set the current points with set_current_points()."
        elif self.interaction_mode == 'desired':
            assert self.desired_pts is not None, "You must set the desired points with set_desired_points()."
        elif self.interaction_mode == 'mean':
            assert self.curr_pts is not None, "You must set the current points with set_current_points()."
            assert self.desired_pts is not None, "You must set the desired points with set_desired_points()."

        # two degrees of freedom: x velocity and z velocity
        if self.control_mode == '2xz':
            # current estimate only
            if self.interaction_mode == 'curr':
                temp_list = []
                for i in range(self.num_pts):
                    temp_list.append(-1.0/self.curr_pts[i][2])
                    temp_list.append(self.curr_pts[i][0]/self.curr_pts[i][2])
                    temp_list.append(0.0)
                    temp_list.append(self.curr_pts[i][1]/self.curr_pts[i][2])
                L_e = np.reshape(temp_list, (2 * self.num_pts, 2))
                self.L_e_est_pinv = np.linalg.pinv(L_e)
            # desired estimate only
            elif self.interaction_mode == 'desired':
                temp_list = []
                for i in range(self.num_pts):
                    temp_list.append(-1.0/self.desired_pts[i][2])
                    temp_list.append(self.desired_pts[i][0]/self.desired_pts[i][2])
                    temp_list.append(0.0)
                    temp_list.append(self.desired_pts[i][1]/self.desired_pts[i][2])
                L_e_desired = np.reshape(temp_list, (2 * self.num_pts, 2))
                self.L_e_est_pinv = np.linalg.pinv(L_e_desired)
            # mean of current and desired
            elif self.interaction_mode == 'mean':
                temp_list = []
                for i in range(self.num_pts):
                    temp_list.append(-1.0/self.curr_pts[i][2])
                    temp_list.append(self.curr_pts[i][0]/self.curr_pts[i][2])
                    temp_list.append(0.0)
                    temp_list.append(self.curr_pts[i][1]/self.curr_pts[i][2])
                L_e = np.reshape(temp_list, (2 * self.num_pts, 2))
                temp_list2 = []
                for i in range(self.num_pts):
                    temp_list2.append(-1.0/self.desired_pts[i][2])
                    temp_list2.append(self.desired_pts[i][0]/self.desired_pts[i][2])
                    temp_list2.append(0.0)
                    temp_list2.append(self.desired_pts[i][1]/self.desired_pts[i][2])
                L_e_desired = np.reshape(temp_list2, (2 * self.num_pts, 2))
                self.L_e_est_pinv = np.linalg.pinv(0.5 * (L_e + L_e_desired))

        # two degrees of freedom: z velocity and y angular velocity
        elif self.control_mode == '2zy':
            # current estimate only
            if self.interaction_mode == 'curr':
                temp_list = []
                for i in range(self.num_pts):
                    temp_list.append(self.curr_pts[i][0]/self.curr_pts[i][2])
                    temp_list.append(-1.0 * (1 + self.curr_pts[i][0]**2))
                    temp_list.append(self.curr_pts[i][1]/self.curr_pts[i][2])
                    temp_list.append(-1.0 * self.curr_pts[i][0] * self.curr_pts[i][1])
                L_e = np.reshape(temp_list, (2 * self.num_pts, 2))
                self.L_e_est_pinv = np.linalg.pinv(L_e)
            # desired estimate only
            elif self.interaction_mode == 'desired':
                temp_list = []
                for i in range(self.num_pts):
                    temp_list.append(self.desired_pts[i][0]/self.desired_pts[i][2])
                    temp_list.append(-1.0 * (1 + self.desired_pts[i][0]**2))
                    temp_list.append(self.desired_pts[i][1]/self.desired_pts[i][2])
                    temp_list.append(-1.0 * self.desired_pts[i][0] * self.desired_pts[i][1])
                L_e_desired = np.reshape(temp_list, (2 * self.num_pts, 2))
                self.L_e_est_pinv = np.linalg.pinv(L_e_desired)
            # mean of current and desired
            elif self.interaction_mode == 'mean':
                temp_list = []
                for i in range(self.num_pts):
                    temp_list.append(self.curr_pts[i][0]/self.curr_pts[i][2])
                    temp_list.append(-1.0 * (1 + self.curr_pts[i][0]**2))
                    temp_list.append(self.curr_pts[i][1]/self.curr_pts[i][2])
                    temp_list.append(-1.0 * self.curr_pts[i][0] * self.curr_pts[i][1])
                L_e = np.reshape(temp_list, (2 * self.num_pts, 2))
                temp_list2 = []
                for i in range(self.num_pts):
                    temp_list2.append(self.desired_pts[i][0]/self.desired_pts[i][2])
                    temp_list2.append(-1.0 * (1 + self.desired_pts[i][0]**2))
                    temp_list2.append(self.desired_pts[i][1]/self.desired_pts[i][2])
                    temp_list2.append(-1.0 * self.desired_pts[i][0] * self.desired_pts[i][1])
                L_e_desired = np.reshape(temp_list2, (2 * self.num_pts, 2))
                self.L_e_est_pinv = np.linalg.pinv(0.5 * (L_e + L_e_desired))

        # four degrees of freedom: x velocity, y velocity, z velocity, and y angular velocity
        elif self.control_mode == '4xyzy':
            # current estimate only
            if self.interaction_mode == 'curr':
                temp_list = []
                for i in range(self.num_pts):
                    temp_list.append(-1.0/self.curr_pts[i][2])
                    temp_list.append(0.0)
                    temp_list.append(self.curr_pts[i][0]/self.curr_pts[i][2])
                    temp_list.append(-1.0 * (1 + self.curr_pts[i][0]**2))
                    temp_list.append(0.0)
                    temp_list.append(-1.0/self.curr_pts[i][2])
                    temp_list.append(self.curr_pts[i][1]/self.curr_pts[i][2])
                    temp_list.append(-1.0 * self.curr_pts[i][0] * self.curr_pts[i][1])
                L_e = np.reshape(temp_list, (2 * self.num_pts, 4))
                self.L_e_est_pinv = np.linalg.pinv(L_e)
            # desired estimate only
            elif self.interaction_mode == 'desired':
                temp_list = []
                for i in range(self.num_pts):
                    temp_list.append(-1.0/self.desired_pts[i][2])
                    temp_list.append(0.0)
                    temp_list.append(self.desired_pts[i][0]/self.desired_pts[i][2])
                    temp_list.append(-1.0 * (1 + self.desired_pts[i][0]**2))
                    temp_list.append(0.0)
                    temp_list.append(-1.0/self.desired_pts[i][2])
                    temp_list.append(self.desired_pts[i][1]/self.desired_pts[i][2])
                    temp_list.append(-1.0 * self.desired_pts[i][0] * self.desired_pts[i][1])
                L_e_desired = np.reshape(temp_list, (2 * self.num_pts, 4))
                self.L_e_est_pinv = np.linalg.pinv(L_e_desired)
            # mean of current and desired
            elif self.interaction_mode == 'mean':
                temp_list = []
                for i in range(self.num_pts):
                    temp_list.append(-1.0/self.curr_pts[i][2])
                    temp_list.append(0.0)
                    temp_list.append(self.curr_pts[i][0]/self.curr_pts[i][2])
                    temp_list.append(-1.0 * (1 + self.curr_pts[i][0]**2))
                    temp_list.append(0.0)
                    temp_list.append(-1.0/self.curr_pts[i][2])
                    temp_list.append(self.curr_pts[i][1]/self.curr_pts[i][2])
                    temp_list.append(-1.0 * self.curr_pts[i][0] * self.curr_pts[i][1])
                L_e = np.reshape(temp_list, (2 * self.num_pts, 4))
                temp_list2 = []
                for i in range(self.num_pts):
                    temp_list2.append(-1.0/self.desired_pts[i][2])
                    temp_list2.append(0.0)
                    temp_list2.append(self.desired_pts[i][0]/self.desired_pts[i][2])
                    temp_list2.append(-1.0 * (1 + self.desired_pts[i][0]**2))
                    temp_list2.append(0.0)
                    temp_list2.append(-1.0/self.desired_pts[i][2])
                    temp_list2.append(self.desired_pts[i][1]/self.desired_pts[i][2])
                    temp_list2.append(-1.0 * self.desired_pts[i][0] * self.desired_pts[i][1])
                L_e_desired = np.reshape(temp_list2, (2 * self.num_pts, 4))
                self.L_e_est_pinv = np.linalg.pinv(0.5 * (L_e + L_e_desired))

    def set_current_points(self, curr_pts: List[Tuple[float, float, float]]):
        '''
        Given a set of current points in the image, set `self.curr_pts` to that set of points and calculate the error vector with `self.calculate_error_vector()` if possible.

        :param curr_pts: A Python list of current points, where each point is a tuple of three floats. The first value in the tuple will be the normalized x coordinate of the point, and should have a value in the range (-1.0, 1.0). The second value in the tuple will be the normalized y coordinate of the point, and should have a value in the range (-1.0, 1.0). The third value in the tuple will be the depth of the point in meters, and should have a value greater than 0.0.
        '''
        assert len(curr_pts) == self.num_pts, f"You must provide {self.num_pts} current points."
        self.curr_pts = curr_pts
        if self.desired_pts != None:
            self.calculate_error_vector()

    def set_desired_points(self, desired_pts: List[Tuple[float, float, float]]):
        '''
        Given a set of desired points in the image, set `self.desired_pts` to that set of points and calculate the error vector with `self.calculate_error_vector()` if possible.

        :param desired_pts: A Python list of desired points, where each point is a tuple of three floats. The first value in the tuple will be the normalized x coordinate of the point, and should have a value in the range (-1.0, 1.0). The second value in the tuple will be the normalized y coordinate of the point, and should have a value in the range (-1.0, 1.0). The third value in the tuple will be the depth of the point in meters, and should have a value greater than 0.0.
        '''
        assert len(desired_pts) == self.num_pts, f"You must provide {self.num_pts} desired points."
        self.desired_pts = desired_pts
        if self.curr_pts is not None:
            self.calculate_error_vector()

    def calculate_error_vector(self):
        '''
        This function is automatically called by either `self.set_current_points()` or `self.set_desired_points()` and should not need to be called by the user, even when providing updated current points.
        '''
        assert self.curr_pts is not None, "You must set the current points with set_current_points()."
        assert self.desired_pts is not None, "You must set the desired points with set_desired_points()."
        errors = []
        for i in range(self.num_pts):
            errors.append(self.curr_pts[i][0] - self.desired_pts[i][0])
            errors.append(self.curr_pts[i][1] - self.desired_pts[i][1])
        self.errs = np.reshape(errors, (2 * self.num_pts, 1))

    def calculate_velocities(self) -> np.ndarray:
        '''
        Calculates the output velocities using the general control equation: `vels = -1 * lambda_matrix * L_e_est_pinv * errs`. Ensure that the lambda matrix, current points, and desired points have been set, and that the interaction matrix has been calculated.

        :return: This function returns a NumPy array containing the velocities. The velocities will be in the order listed in the associated control mode.
        '''
        assert self.lambda_matrix is not None, "You must set the lambda matrix with set_lambda_matrix()."
        assert self.L_e_est_pinv is not None, "You must set the error interaction matrix estimate with calculate_interaction_matrix()."
        assert self.errs is not None, "You must set the errors by setting both the current and desired points, or by manually calling calculate_error_vector()."
        self.vels = -1.0 * (self.lambda_matrix @ self.L_e_est_pinv @ self.errs)
        return list(self.vels)

# testing
if __name__ == '__main__':
    pts_2_center_near = [(-0.5, -0.5, 1.0), (0.5, 0.5, 1.0)]
    pts_2_center_far = [(-0.2, -0.2, 5.0), (0.2, 0.2, 5.0)]
    pts_2_left = [(-0.75, -0.5, 1.0), (0.25, 0.5, 1.0)]
    pts_2_right = [(-0.25, -0.5, 1.0), (0.75, 0.5, 1.0)]

    pts_4_center_near = [(-0.5, -0.5, 1.0), (0.5, -0.5, 1.0), (-0.5, 0.5, 1.0), (0.5, 0.5, 1.0)]
    pts_4_center_far = [(-0.2, -0.2, 5.0), (0.2, -0.2, 5.0), (-0.2, 0.2, 5.0), (0.2, 0.2, 5.0)]
    pts_4_left = [(-0.75, -0.5, 1.0), (0.25, -0.5, 1.0), (-0.75, 0.5, 1.0), (0.25, 0.5, 1.0)]
    pts_4_right = [(-0.25, -0.5, 1.0), (0.75, -0.5, 1.0), (-0.25, 0.5, 1.0), (0.75, 0.5, 1.0)]
    pts_4_up = [(-0.5, -0.75, 1.0), (0.5, -0.75, 1.0), (-0.5, 0.25, 1.0), (0.5, 0.25, 1.0)]
    pts_4_down = [(-0.5, -0.25, 1.0), (0.5, -0.25, 1.0), (-0.5, 0.75, 1.0), (0.5, 0.75, 1.0)]
    pts_4_left_turned = [(-0.75, -0.75, 1.0), (-0.25, -0.5, 2.0), (-0.75, 0.75, 1.0), (-0.25, 0.5, 2.0)]
    pts_4_right_turned = [(0.25, -0.5, 2.0), (0.75, -0.75, 1.0), (0.25, 0.5, 2.0), (0.75, 0.75, 1.0)]
    
    print("### TEST 1: 2DOF xz controller, curr mode, move forward ###")
    controller = IBVS_Controller(control_mode='2xz', interaction_mode='curr', num_pts=2)
    controller.set_lambda_matrix(lambdas=[5.0, 5.0])
    controller.set_desired_points(desired_pts=pts_2_center_near)
    controller.set_current_points(curr_pts=pts_2_center_far)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")

    print("\n### TEST 2: 2DOF xz controller, curr mode, move backward ###")
    controller.set_desired_points(desired_pts=pts_2_center_far)
    controller.set_current_points(curr_pts=pts_2_center_near)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")
    
    print("\n### TEST 3: 2DOF xz controller, curr mode, move right ###")
    controller.set_desired_points(desired_pts=pts_2_left)
    controller.set_current_points(curr_pts=pts_2_right)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")
    
    print("\n### TEST 4: 2DOF xz controller, curr mode, move left ###")
    controller.set_desired_points(desired_pts=pts_2_right)
    controller.set_current_points(curr_pts=pts_2_left)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")
    
    print("\n### TEST 5: 2DOF xz controller, desired mode, move forward ###")
    controller = IBVS_Controller(control_mode='2xz', interaction_mode='desired', num_pts=2)
    controller.set_lambda_matrix(lambdas=[5.0, 5.0])
    controller.set_desired_points(desired_pts=pts_2_center_near)
    controller.set_current_points(curr_pts=pts_2_center_far)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")

    print("\n### TEST 6: 2DOF xz controller, desired mode, move backward ###")
    controller.set_desired_points(desired_pts=pts_2_center_far)
    controller.set_current_points(curr_pts=pts_2_center_near)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")

    print("\n### TEST 7: 2DOF xz controller, desired mode, move right ###")
    controller.set_desired_points(desired_pts=pts_2_left)
    controller.set_current_points(curr_pts=pts_2_right)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")

    print("\n### TEST 8: 2DOF xz controller, desired mode, move left ###")
    controller.set_desired_points(desired_pts=pts_2_right)
    controller.set_current_points(curr_pts=pts_2_left)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")

    print("\n### TEST 9: 2DOF xz controller, mean mode, move forward ###")
    controller = IBVS_Controller(control_mode='2xz', interaction_mode='mean', num_pts=2)
    controller.set_lambda_matrix(lambdas=[5.0, 5.0])
    controller.set_desired_points(desired_pts=pts_2_center_near)
    controller.set_current_points(curr_pts=pts_2_center_far)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")

    print("\n### TEST 10: 2DOF xz controller, mean mode, move backward ###")
    controller.set_desired_points(desired_pts=pts_2_center_far)
    controller.set_current_points(curr_pts=pts_2_center_near)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")

    print("\n### TEST 11: 2DOF xz controller, mean mode, move right ###")
    controller.set_desired_points(desired_pts=pts_2_left)
    controller.set_current_points(curr_pts=pts_2_right)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")

    print("\n### TEST 12: 2DOF xz controller, mean mode, move left ###")
    controller.set_desired_points(desired_pts=pts_2_right)
    controller.set_current_points(curr_pts=pts_2_left)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")



    print("\n### TEST 13: 2DOF zy controller, curr mode, move forward ###")
    controller = IBVS_Controller(control_mode='2zy', interaction_mode='curr', num_pts=2)
    controller.set_lambda_matrix(lambdas=[5.0, 5.0])
    controller.set_desired_points(desired_pts=pts_2_center_near)
    controller.set_current_points(curr_pts=pts_2_center_far)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")

    print("\n### TEST 14: 2DOF zy controller, curr mode, move backward ###")
    controller.set_desired_points(desired_pts=pts_2_center_far)
    controller.set_current_points(curr_pts=pts_2_center_near)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")

    print("\n### TEST 15: 2DOF zy controller, curr mode, turn right ###")
    controller.set_desired_points(desired_pts=pts_2_left)
    controller.set_current_points(curr_pts=pts_2_right)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")

    print("\n### TEST 16: 2DOF zy controller, curr mode, turn left ###")
    controller.set_desired_points(desired_pts=pts_2_right)
    controller.set_current_points(curr_pts=pts_2_left)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")

    print("\n### TEST 17: 2DOF zy controller, desired mode, move forward ###")
    controller = IBVS_Controller(control_mode='2zy', interaction_mode='desired', num_pts=2)
    controller.set_lambda_matrix(lambdas=[5.0, 5.0])
    controller.set_desired_points(desired_pts=pts_2_center_near)
    controller.set_current_points(curr_pts=pts_2_center_far)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")

    print("\n### TEST 18: 2DOF zy controller, desired mode, move backward ###")
    controller.set_desired_points(desired_pts=pts_2_center_far)
    controller.set_current_points(curr_pts=pts_2_center_near)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")

    print("\n### TEST 19: 2DOF zy controller, desired mode, turn right ###")
    controller.set_desired_points(desired_pts=pts_2_left)
    controller.set_current_points(curr_pts=pts_2_right)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")

    print("\n### TEST 20: 2DOF zy controller, desired mode, turn left ###")
    controller.set_desired_points(desired_pts=pts_2_right)
    controller.set_current_points(curr_pts=pts_2_left)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")

    print("\n### TEST 21: 2DOF zy controller, mean mode, move forward ###")
    controller = IBVS_Controller(control_mode='2zy', interaction_mode='mean', num_pts=2)
    controller.set_lambda_matrix(lambdas=[5.0, 5.0])
    controller.set_desired_points(desired_pts=pts_2_center_near)
    controller.set_current_points(curr_pts=pts_2_center_far)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")

    print("\n### TEST 22: 2DOF zy controller, mean mode, move backward ###")
    controller.set_desired_points(desired_pts=pts_2_center_far)
    controller.set_current_points(curr_pts=pts_2_center_near)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")

    print("\n### TEST 23: 2DOF zy controller, mean mode, turn right ###")
    controller.set_desired_points(desired_pts=pts_2_left)
    controller.set_current_points(curr_pts=pts_2_right)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")

    print("\n### TEST 24: 2DOF zy controller, mean mode, turn left ###")
    controller.set_desired_points(desired_pts=pts_2_right)
    controller.set_current_points(curr_pts=pts_2_left)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")



    print("\n### TEST 25: 4DOF xyzy controller, curr mode, move forward ###")
    controller = IBVS_Controller(control_mode='4xyzy', interaction_mode='curr', num_pts=4)
    controller.set_lambda_matrix(lambdas=[5.0, 5.0, 5.0, 5.0])
    controller.set_desired_points(desired_pts=pts_4_center_near)
    controller.set_current_points(curr_pts=pts_4_center_far)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")

    print("\n### TEST 26: 4DOF xyzy controller, curr mode, move backward ###")
    controller.set_desired_points(desired_pts=pts_4_center_far)
    controller.set_current_points(curr_pts=pts_4_center_near)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")

    print("\n### TEST 27: 4DOF xyzy controller, curr mode, move right ###")
    controller.set_desired_points(desired_pts=pts_4_left)
    controller.set_current_points(curr_pts=pts_4_right)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")

    print("\n### TEST 28: 4DOF xyzy controller, curr mode, move left ###")
    controller.set_desired_points(desired_pts=pts_4_right)
    controller.set_current_points(curr_pts=pts_4_left)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")

    print("\n### TEST 29: 4DOF xyzy controller, curr mode, move down ###")
    controller.set_desired_points(desired_pts=pts_4_up)
    controller.set_current_points(curr_pts=pts_4_down)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")

    print("\n### TEST 30: 4DOF xyzy controller, curr mode, move up ###")
    controller.set_desired_points(desired_pts=pts_4_down)
    controller.set_current_points(curr_pts=pts_4_up)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")

    print("\n### TEST 31: 4DOF xyzy controller, curr mode, turn right ###")
    controller.set_desired_points(desired_pts=pts_4_left_turned)
    controller.set_current_points(curr_pts=pts_4_right_turned)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")

    print("\n### TEST 32: 4DOF xyzy controller, curr mode, turn left ###")
    controller.set_desired_points(desired_pts=pts_4_right_turned)
    controller.set_current_points(curr_pts=pts_4_left_turned)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")

    print("\n### TEST 33: 4DOF xyzy controller, desired mode, move forward ###")
    controller = IBVS_Controller(control_mode='4xyzy', interaction_mode='desired', num_pts=4)
    controller.set_lambda_matrix(lambdas=[5.0, 5.0, 5.0, 5.0])
    controller.set_desired_points(desired_pts=pts_4_center_near)
    controller.set_current_points(curr_pts=pts_4_center_far)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")

    print("\n### TEST 34: 4DOF xyzy controller, desired mode, move backward ###")
    controller.set_desired_points(desired_pts=pts_4_center_far)
    controller.set_current_points(curr_pts=pts_4_center_near)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")

    print("\n### TEST 35: 4DOF xyzy controller, desired mode, move right ###")
    controller.set_desired_points(desired_pts=pts_4_left)
    controller.set_current_points(curr_pts=pts_4_right)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")

    print("\n### TEST 36: 4DOF xyzy controller, desired mode, move left ###")
    controller.set_desired_points(desired_pts=pts_4_right)
    controller.set_current_points(curr_pts=pts_4_left)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")

    print("\n### TEST 37: 4DOF xyzy controller, desired mode, move down ###")
    controller.set_desired_points(desired_pts=pts_4_up)
    controller.set_current_points(curr_pts=pts_4_down)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")

    print("\n### TEST 38: 4DOF xyzy controller, desired mode, move up ###")
    controller.set_desired_points(desired_pts=pts_4_down)
    controller.set_current_points(curr_pts=pts_4_up)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")

    print("\n### TEST 39: 4DOF xyzy controller, desired mode, turn right ###")
    controller.set_desired_points(desired_pts=pts_4_left_turned)
    controller.set_current_points(curr_pts=pts_4_right_turned)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")

    print("\n### TEST 40: 4DOF xyzy controller, desired mode, turn left ###")
    controller.set_desired_points(desired_pts=pts_4_right_turned)
    controller.set_current_points(curr_pts=pts_4_left_turned)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")

    print("\n### TEST 41: 4DOF xyzy controller, mean mode, move forward ###")
    controller = IBVS_Controller(control_mode='4xyzy', interaction_mode='mean', num_pts=4)
    controller.set_lambda_matrix(lambdas=[5.0, 5.0, 5.0, 5.0])
    controller.set_desired_points(desired_pts=pts_4_center_near)
    controller.set_current_points(curr_pts=pts_4_center_far)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")

    print("\n### TEST 42: 4DOF xyzy controller, mean mode, move backward ###")
    controller.set_desired_points(desired_pts=pts_4_center_far)
    controller.set_current_points(curr_pts=pts_4_center_near)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")

    print("\n### TEST 43: 4DOF xyzy controller, mean mode, move right ###")
    controller.set_desired_points(desired_pts=pts_4_left)
    controller.set_current_points(curr_pts=pts_4_right)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")

    print("\n### TEST 44: 4DOF xyzy controller, mean mode, move left ###")
    controller.set_desired_points(desired_pts=pts_4_right)
    controller.set_current_points(curr_pts=pts_4_left)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")

    print("\n### TEST 45: 4DOF xyzy controller, mean mode, move down ###")
    controller.set_desired_points(desired_pts=pts_4_up)
    controller.set_current_points(curr_pts=pts_4_down)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")

    print("\n### TEST 46: 4DOF xyzy controller, mean mode, move up ###")
    controller.set_desired_points(desired_pts=pts_4_down)
    controller.set_current_points(curr_pts=pts_4_up)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")

    print("\n### TEST 47: 4DOF xyzy controller, mean mode, turn right ###")
    controller.set_desired_points(desired_pts=pts_4_left_turned)
    controller.set_current_points(curr_pts=pts_4_right_turned)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")

    print("\n### TEST 48: 4DOF xyzy controller, mean mode, turn left ###")
    controller.set_desired_points(desired_pts=pts_4_right_turned)
    controller.set_current_points(curr_pts=pts_4_left_turned)
    controller.calculate_interaction_matrix()
    vels = controller.calculate_velocities()
    print(f"output velocities: {vels}")
    