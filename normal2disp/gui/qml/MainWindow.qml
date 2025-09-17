import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtQuick.Window 2.15

import "." as Components

ApplicationWindow {
    id: window
    width: 1280
    height: 720
    visible: true
    title: "normal2disp"
    color: theme.background
    minimumWidth: 1024
    minimumHeight: 640

    property var backend: appBackend

    Components.Theme { id: theme }

    ColumnLayout {
        anchors.fill: parent
        spacing: theme.spacing
        anchors.margins: theme.padding

        RowLayout {
            id: mainRow
            Layout.fillWidth: true
            Layout.fillHeight: true
            spacing: theme.spacing

            Components.LeftPanel {
                id: leftPanel
                theme: theme
                backend: backend
                Layout.preferredWidth: 340
                Layout.maximumWidth: 360
                Layout.fillHeight: true
            }

            Components.Viewport {
                id: viewport
                theme: theme
                Layout.fillWidth: true
                Layout.fillHeight: true
                Layout.preferredWidth: 640
            }

            Components.RightPane {
                id: rightPane
                theme: theme
                backend: backend
                Layout.preferredWidth: 320
                Layout.maximumWidth: 340
                Layout.fillHeight: true
            }
        }

        Components.StatusBar {
            id: statusBar
            theme: theme
            backend: backend
            Layout.fillWidth: true
            Layout.preferredHeight: 200
        }
    }
}
