#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Activity grouping definitions for BPI 2019 dataset
"""

# Define activity groups
ACTIVITY_GROUPS = {
    'create_orders': [
        'Create Purchase Order Item',
        'Create Purchase Requisition Item',
        'Record Service Entry Sheet',
        'Record Goods Receipt'
    ],
    'change_orders': [
        'Change Price',
        'Change Quantity',
        'Change Delivery Indicator',
        'Change Approval for Purchase Order'
    ],
    'approve_cancel': [
        'SRM: Awaiting Approval',
        'Send for Approval',
        'Approve Purchase Order',
        'Cancel Goods Receipt',
        'Cancel Invoice Receipt',
        'Remove Payment Block',
        'Block Purchase Order Item'
    ],
    'record_receipts': [
        'Record Invoice Receipt',
        'Record Service Entry Sheet',
        'Record Goods Receipt',
        'Clear Invoice'
    ],
    'vendor_actions': [
        'Vendor creates invoice',
        'Receive Order Confirmation',
        'Update Order Confirmation'
    ],
    'system_interactions': [
        'SRM: Complete',
        'SRM: In Transfer to Execution Syst.',
        'SRM: Document Completed',
        'SRM: Ordered',
        'SRM: Transfer Failed',
        'SRM: Transaction completed',
        'Delete Purchase Order Item',
        'Release Purchase Order'
    ]
}

def get_activity_group(activity_name):
    """Get the group for a given activity"""
    for group, activities in ACTIVITY_GROUPS.items():
        if activity_name in activities:
            return group
    return 'other'

def get_group_id(group_name):
    """Get numerical ID for a group"""
    groups = list(ACTIVITY_GROUPS.keys()) + ['other']
    return groups.index(group_name)

def get_num_groups():
    """Get total number of groups (including 'other')"""
    return len(ACTIVITY_GROUPS) + 1  # +1 for 'other' group 